import os
import json
import cv2
import numpy as np
from typing import List
import re
import urllib.parse
import base64

# Mapping of class IDs to class names
LABELS_MAPPING = {
    0: "cable",

}

def mapping_class(class_name) -> int:
    """Map a class name to its corresponding class ID. Accepts string or list."""
    if isinstance(class_name, list):
        class_name = class_name[0] if class_name else ""
    try:
        return list(LABELS_MAPPING.keys())[list(LABELS_MAPPING.values()).index(class_name)]
    except ValueError:
        raise ValueError(f"Class name '{class_name}' not found in LABELS_MAPPING")

class InputStream:
    """Helper class to read bits from a binary string."""
    def __init__(self, data: str):
        self.data = data
        self.i = 0
    
    def read(self, size: int) -> int:
        """Read specified number of bits and convert to integer."""
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)

def access_bit(data: bytes, num: int) -> int:
    """Extract a single bit from a byte array at the specified position."""
    base = num // 8
    shift = 7 - (num % 8)
    return (data[base] & (1 << shift)) >> shift

def bytes2bit(data: bytes) -> str:
    """Convert byte array to a binary string."""
    return "".join(str(access_bit(data, i)) for i in range(len(data) * 8))

def decode_mask_from_rle(rle, height, width):
    """
    返回二值 mask (uint8, 0/255)，支持：
      - list/tuple/ndarray: 原始字节数组 (RGBA or grayscale) 或按像素值数组 或 LabelStudio 的 packed RLE list
      - str: base64 编码的图像数据
      - dict with 'counts': COCO RLE (if pycocotools 可用)
    """
    # COCO RLE dict
    if isinstance(rle, dict) and "counts" in rle:
        try:
            from pycocotools import mask as cocomask
            m = cocomask.decode(rle)
            if m.ndim == 3:
                m = m[:, :, 0]
            return (m > 0).astype(np.uint8) * 255
        except Exception:
            raise

    # base64 string
    if isinstance(rle, str):
        try:
            b = base64.b64decode(rle)
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("base64 decode -> not an image")
            if img.ndim == 3 and img.shape[2] == 4:
                return img[:, :, 3]
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            return mask
        except Exception:
            pass

    # Label Studio packed RLE list (bitstream format)
    try:
        # Accept list/tuple/ndarray of ints as packed rle
        if isinstance(rle, (list, tuple, np.ndarray)) and len(rle) < height * width:
            bitstr = bytes2bit(bytes(rle))
            rle_input = InputStream(bitstr)

            num = rle_input.read(32)
            word_size = rle_input.read(5) + 1
            rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]

            out = np.zeros(num, dtype=np.uint8)
            i = 0
            while i < num:
                x = rle_input.read(1)
                j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
                if x:
                    val = rle_input.read(word_size)
                    out[i:j] = val
                    i = j
                else:
                    while i < j:
                        out[i] = rle_input.read(word_size)
                        i += 1

            image = np.reshape(out, [height, width, 4])[:, :, 3]
            _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
            return mask
    except Exception:
        pass

    # list/bytes/ndarray (fallbacks)
    if isinstance(rle, (list, tuple, np.ndarray, bytes, bytearray)):
        arr = np.frombuffer(bytes(rle), dtype=np.uint8) if isinstance(rle, (bytes, bytearray)) else np.array(rle, dtype=np.uint8)
        # RGBA bytes
        if arr.size == height * width * 4:
            img = arr.reshape((height, width, 4))
            return img[:, :, 3]
        # per-pixel mask
        if arr.size == height * width:
            mask = arr.reshape((height, width))
            return (mask > 0).astype(np.uint8) * 255
        # 尝试当作编码图像字节（PNG/JPEG）
        try:
            img = cv2.imdecode(np.frombuffer(bytes(arr), np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 3 and img.shape[2] == 4:
                    return img[:, :, 3]
                gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                return mask
        except Exception:
            pass

    raise ValueError("Unsupported brush RLE/format")

def brush_to_yolo(rle, height, width, min_area=200, epsilon_factor=0.01):
    """
    返回 List[List[float]]：每个内层列表为 [x1,y1,x2,y2,...]（均已归一化到 0..1）
    """
    mask = decode_mask_from_rle(rle, height, width)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        eps = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = []
        for (x, y) in approx.reshape(-1, 2):
            pts.extend([float(x) / width, float(y) / height])
        if len(pts) >= 6:
            polygons.append(pts)
    return polygons

def polygon_to_yolo(points):
    """
    处理多种输入格式：
      - [{'x':..,'y':..}, ...]
      - [[x,y], ...]
    自动判断单位：如果值 <=1 视为已归一化；<=100 视为百分比；否则保留（需要额外宽高时再除以像素）
    """
    out = []
    for p in points:
        if isinstance(p, dict):
            x, y = p.get("x", 0), p.get("y", 0)
        else:
            x, y = p
        def norm(v):
            if v <= 1.0:
                return v
            if v <= 100.0:
                return v / 100.0
            return v  # 可能为像素，需要调用处知道 width/height 再处理
        out.extend([norm(float(x)), norm(float(y))])
    return out

def json_to_yolo(input_file: str, output_dir: str) -> None:
    with open(input_file, "r") as f:
        data = json.load(f)
    
    skipped_labels = []
    for task in data:
        raw_image = task["data"].get("image", "")
        # Try to extract the 'd' query parameter (Label Studio local file reference) and decode it.
        parsed = urllib.parse.urlparse(raw_image)
        qs = urllib.parse.parse_qs(parsed.query)
        if 'd' in qs and qs['d']:
            decoded = urllib.parse.unquote(qs['d'][0])
        else:
            # Fallback: decode the whole URL or path
            decoded = urllib.parse.unquote(raw_image)
        # Normalize separators and take the basename as image filename
        decoded_norm = decoded.replace('\\', '/').rstrip('/')
        image_basename = os.path.basename(decoded_norm.split('/')[-1])
        image_name, _ = os.path.splitext(image_basename)

        def sanitize_filename(name: str) -> str:
            name = name.split("?")[0]
            name = urllib.parse.unquote(name)
            name = os.path.basename(name)
            # replace invalid Windows filename chars with underscore, but preserve spaces
            name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
            name = name.strip().rstrip('. ')
            # keep spaces (do not replace whitespace with underscores)
            name = name
            return name

        safe_name = sanitize_filename(image_name)
        if not safe_name:
            safe_name = f"task_{task.get('id') or 'unknown'}"
        polygons = []
        
        for annotation in task.get("annotations", []):
            for item in annotation["result"]:
                height = item["original_height"]
                width = item["original_width"]
                
                if item.get("type") == "brushlabels":
                    polys = brush_to_yolo(item["value"]["rle"], height, width)
                    for p in polys:
                        polygons.append({
                            "points": p,
                            "class": item["value"]["brushlabels"]
                        })
                    continue
                elif item.get("type") == "polygonlabels":
                    polygon = {
                        "points": polygon_to_yolo(item["value"]["points"]),
                        "class": item["value"]["polygonlabels"]
                    }
                else:
                    skipped_labels.append({
                        "task_id": task.get("id"),
                        "type": item.get("type"),
                        "id": item.get("id")
                    })
                    continue
                polygons.append(polygon)

        if not polygons:
            print(f"Skipping: {safe_name} (no valid polygons)")
            continue

        out_path = os.path.join(output_dir, f"{safe_name}.txt")
        try:
            with open(out_path, "w", newline='') as f:
                for polygon in polygons:
                    pts = polygon["points"]
                    cls = polygon["class"]
                    class_label = cls[0] if isinstance(cls, list) else cls
                    try:
                        class_id = mapping_class(class_label)
                    except ValueError as e:
                        skipped_labels.append({
                            "task_id": task.get("id"),
                            "type": "class_mapping",
                            "detail": str(e)
                        })
                        continue
                    f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in pts)}\n")
            print(f"Converted: {safe_name}.txt")
        except OSError as e:
            print(f"Failed to write file for image '{image_name}' -> '{out_path}': {e}")
    
    print("Conversion completed.")
    if skipped_labels:
        print(f"Skipped labels: {skipped_labels}")

if __name__ == "__main__":
    input_file = "brush.json"
    output_dir = "tunnel/labels"
    os.makedirs(output_dir, exist_ok=True)
    json_to_yolo(input_file, output_dir)