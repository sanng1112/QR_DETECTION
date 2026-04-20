"""
    python main.py --data qr/public_train.csv
    python main.py --data qr/public_train.csv --gt qr/output_train.csv   # kèm đánh giá
    python main.py --eval-only --pred output.csv --gt qr/output_train.csv # chỉ đánh giá
"""

import argparse
import os
import time
import csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


DEFAULT_MODEL  = "./runs/obb/QR_OBB_Training/run_v1/weights/best.pt"
DEFAULT_OUTPUT = "output.csv"
INFER_SIZE     = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
DEVICE         = "cpu"
COORD_COLS     = ["x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"]


def _clahe_enhance(gray: np.ndarray) -> np.ndarray:
    """CLAHE trên kênh L của LAB — tăng tương phản cục bộ."""
    lab = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)[:, :, 0]


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask nhẹ để làm nét module QR."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Fast Non-Local Means — khử nhiễu giữ cạnh."""
    return cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)


def enhance_for_decode(gray: np.ndarray) -> np.ndarray:
    """
    Pipeline tăng cường ảnh grayscale trước khi decode.
    Thứ tự: denoise → CLAHE → sharpen
    """
    out = _denoise(gray)
    out = _clahe_enhance(out)
    out = _sharpen(out)
    return out


def preprocess_image(img: np.ndarray, target_size: int = INFER_SIZE):
    """
    Letterbox resize: giữ tỷ lệ, pad bằng xám 114.
    Trả về: (canvas_BGR, scale, pad_left, pad_top)
    """
    h, w  = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized  = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas       = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_left     = (target_size - new_w) // 2
    pad_top      = (target_size - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_resized
    return canvas, scale, pad_left, pad_top


def undo_preprocess(pts_on_canvas: np.ndarray,
                    scale: float, pad_left: int, pad_top: int) -> np.ndarray:
    """Chuyển tọa độ từ canvas letterbox về ảnh gốc."""
    pts = pts_on_canvas.copy().astype(np.float32)
    pts[:, 0] = (pts[:, 0] - pad_left) / scale
    pts[:, 1] = (pts[:, 1] - pad_top)  / scale
    return pts


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 góc: top-left → top-right → bottom-right → bottom-left."""
    pts     = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s       = pts.sum(axis=1)
    diff    = np.diff(pts, axis=1).flatten()
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def _try_cv2(img_gray: np.ndarray) -> str:
    """Backend cv2.QRCodeDetectorAruco (mới hơn, ổn định hơn QRCodeDetector)."""
    try:
        det = cv2.QRCodeDetectorAruco()
        data, _, _ = det.detectAndDecode(img_gray)
        return data if data else ""
    except Exception:
        pass
    try:
        det = cv2.QRCodeDetector()
        data, _, _ = det.detectAndDecode(img_gray)
        return data if data else ""
    except Exception:
        return ""


def _try_pyzbar(img_gray: np.ndarray) -> str:
    """Backend pyzbar (nếu cài đặt)."""
    try:
        from pyzbar.pyzbar import decode as pyz_decode
        from pyzbar.pyzbar import ZBarSymbol
        results = pyz_decode(img_gray, symbols=[ZBarSymbol.QRCODE])
        if results:
            return results[0].data.decode("utf-8", errors="replace")
    except ImportError:
        pass
    except Exception:
        pass
    return ""


def _try_zxing(img_gray: np.ndarray) -> str:
    """Backend zxing-cpp (nếu cài đặt)."""
    try:
        import zxingcpp
        results = zxingcpp.read_barcodes(img_gray)
        for r in results:
            if r.valid and r.format.name == "QRCode":
                return r.text
    except ImportError:
        pass
    except Exception:
        pass
    return ""

def _warp_qr(img_orig: np.ndarray, pts: np.ndarray, out_size: int = 256) -> np.ndarray:
    """
    Perspective warp 4 điểm về ảnh vuông out_size × out_size.
    Trả về ảnh grayscale đã warp.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    dst = np.array([[0, 0],
                    [out_size - 1, 0],
                    [out_size - 1, out_size - 1],
                    [0, out_size - 1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(pts, dst)
    warped_bgr = cv2.warpPerspective(img_orig, M, (out_size, out_size),
                                     flags=cv2.INTER_CUBIC)
    return cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)



def _binarize_variants(gray: np.ndarray) -> list:
    variants = [gray]

    # CLAHE + sharpen
    enhanced = _clahe_enhance(gray)
    enhanced = _sharpen(enhanced)
    variants.append(enhanced)

    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # Adaptive mean
    adap_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10)
    variants.append(adap_mean)

    # Adaptive Gaussian
    adap_gauss = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10)
    variants.append(adap_gauss)

    # Otsu sau CLAHE
    _, otsu_clahe = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu_clahe)

    # Đảo màu
    variants.append(cv2.bitwise_not(otsu))

    return variants


def _rotation_variants(gray: np.ndarray) -> list:
    """Thêm 3 bản xoay (90°, 180°, 270°) của ảnh."""
    return [
        gray,
        cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(gray, cv2.ROTATE_180),
        cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]


def _scale_up(gray: np.ndarray, min_side: int = 64,
              target: int = 256) -> np.ndarray:
    """Nếu QR quá nhỏ, scale up để decoder nhận diện tốt hơn."""
    h, w = gray.shape[:2]
    if min(h, w) < min_side:
        scale = target / min(h, w)
        return cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return gray

_BACKENDS = [_try_cv2, _try_pyzbar, _try_zxing]


def _run_cascade(gray: np.ndarray) -> str:
    """
    Thử tất cả (backend × biến thể ảnh × xoay).
    Early-exit ngay khi có kết quả.
    """
    gray = _scale_up(gray)

    for variant in _binarize_variants(gray):
        for rotated in _rotation_variants(variant):
            for backend in _BACKENDS:
                result = backend(rotated)
                if result:
                    return result
    return ""

def decode_qr(img_orig: np.ndarray, pts: np.ndarray) -> str:
    """
    Giải mã nội dung QR từ ảnh gốc và 4 điểm góc (đã ordered).

    Chiến lược (từ nhanh đến chậm):
      Pass 1 — Warp perspective → cascade biến thể (ảnh chuẩn nhất)
      Pass 2 — Crop bbox mở rộng → cascade biến thể (fallback)
    """
    try:
        warp_sizes = [256, 512]  # thử 2 kích thước
        for sz in warp_sizes:
            warped = _warp_qr(img_orig, pts, out_size=sz)
            result = _run_cascade(warped)
            if result:
                return result
    except Exception:
        pass

    # ── Pass 2: crop bbox thô (fallback khi warp bị biến dạng) ───
    try:
        margin = 8
        x_min = max(0, int(pts[:, 0].min()) - margin)
        y_min = max(0, int(pts[:, 1].min()) - margin)
        x_max = min(img_orig.shape[1], int(pts[:, 0].max()) + margin)
        y_max = min(img_orig.shape[0], int(pts[:, 1].max()) + margin)
        crop_bgr = img_orig[y_min:y_max, x_min:x_max]
        if crop_bgr.size == 0:
            return ""
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        result = _run_cascade(crop_gray)
        if result:
            return result
    except Exception:
        pass

    return ""

def detect_qr_in_image(model, img_path: str, decode: bool = True):
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print(f"  [!] Không đọc được ảnh: {img_path}")
        return []

    img_canvas, scale, pad_left, pad_top = preprocess_image(img_orig, INFER_SIZE)

    results = model.predict(
        source=img_canvas, imgsz=INFER_SIZE,
        conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
        device=DEVICE, verbose=False,
    )

    obb = results[0].obb
    if obb is None or len(obb) == 0:
        return []

    pts_all = obb.xyxyxyxy.cpu().numpy()
    confs   = obb.conf.cpu().numpy()
    order   = np.argsort(confs)[::-1]

    detections = []
    for qr_idx, i in enumerate(order):
        pts_orig    = undo_preprocess(pts_all[i], scale, pad_left, pad_top)
        pts_ordered = order_points(pts_orig)
        content     = decode_qr(img_orig, pts_ordered) if decode else ""
        detections.append({
            "qr_index": qr_idx,
            "x0": round(float(pts_ordered[0, 0]), 2),
            "y0": round(float(pts_ordered[0, 1]), 2),
            "x1": round(float(pts_ordered[1, 0]), 2),
            "y1": round(float(pts_ordered[1, 1]), 2),
            "x2": round(float(pts_ordered[2, 0]), 2),
            "y2": round(float(pts_ordered[2, 1]), 2),
            "x3": round(float(pts_ordered[3, 0]), 2),
            "y3": round(float(pts_ordered[3, 1]), 2),
            "content": content,
        })

    return detections

def _polygon_area(poly: np.ndarray) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_polygon_by_halfplane(poly: list,
                                a: np.ndarray, b: np.ndarray) -> list:
    if not poly:
        return []
    output = []
    n = len(poly)
    for i in range(n):
        cur  = np.array(poly[i],           dtype=np.float64)
        prev = np.array(poly[(i - 1) % n], dtype=np.float64)

        def inside(p):
            return ((b[0] - a[0]) * (p[1] - a[1])
                    - (b[1] - a[1]) * (p[0] - a[0])) >= 0

        if inside(cur):
            if not inside(prev):
                d1    = cur - prev
                d2    = b   - a
                denom = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(denom) > 1e-10:
                    t = ((a[0] - prev[0]) * d2[1]
                         - (a[1] - prev[1]) * d2[0]) / denom
                    output.append((prev + t * d1).tolist())
            output.append(cur.tolist())
        elif inside(prev):
            d1    = cur - prev
            d2    = b   - a
            denom = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(denom) > 1e-10:
                t = ((a[0] - prev[0]) * d2[1]
                     - (a[1] - prev[1]) * d2[0]) / denom
                output.append((prev + t * d1).tolist())
    return output


def polygon_iou(pts_p: np.ndarray, pts_g: np.ndarray) -> float:
    pts_p = pts_p.reshape(4, 2).astype(np.float64)
    pts_g = pts_g.reshape(4, 2).astype(np.float64)
    area_p = _polygon_area(pts_p)
    area_g = _polygon_area(pts_g)
    if area_p < 1e-6 or area_g < 1e-6:
        return 0.0
    clipped = pts_p.tolist()
    n = len(pts_g)
    for i in range(n):
        a = pts_g[i]
        b = pts_g[(i + 1) % n]
        clipped = _clip_polygon_by_halfplane(clipped, a, b)
        if not clipped:
            return 0.0
    inter = _polygon_area(np.array(clipped))
    union = area_p + area_g - inter
    return float(inter / union) if union > 1e-6 else 0.0


def _load_boxes(df: pd.DataFrame) -> dict:
    result = {}
    for _, row in df.iterrows():
        img_id  = str(row["image_id"])
        qr_idx  = str(row.get("qr_index", "")).strip()
        if pd.isna(row.get("qr_index", float("nan"))) or qr_idx == "":
            result.setdefault(img_id, [])
            continue
        try:
            coords = [float(row[c]) for c in COORD_COLS]
        except (ValueError, KeyError):
            result.setdefault(img_id, [])
            continue
        box = np.array(coords, dtype=np.float32).reshape(4, 2)
        result.setdefault(img_id, []).append(box)
    return result


def evaluate(pred_csv: str, gt_csv: str,
             iou_threshold: float = 0.5) -> dict:
    pred_df    = pd.read_csv(pred_csv)
    gt_df      = pd.read_csv(gt_csv)
    pred_boxes = _load_boxes(pred_df)
    gt_boxes   = _load_boxes(gt_df)

    all_ids = set(gt_boxes.keys()) | set(pred_boxes.keys())
    total_tp = total_fp = total_fn = 0
    iou_tp_list = []
    per_image   = []

    for img_id in sorted(all_ids):
        preds = pred_boxes.get(img_id, [])
        gts   = gt_boxes.get(img_id, [])

        iou_matrix = np.zeros((len(preds), len(gts)), dtype=np.float32)
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                iou_matrix[pi, gi] = polygon_iou(p, g)

        matched_pred = set()
        matched_gt   = set()
        tp_pairs     = []

        if len(preds) > 0 and len(gts) > 0:
            flat_order = np.argsort(iou_matrix, axis=None)[::-1]
            for flat_idx in flat_order:
                pi, gi = divmod(int(flat_idx), len(gts))
                iou_val = float(iou_matrix[pi, gi])
                if iou_val < iou_threshold:
                    break
                if pi in matched_pred or gi in matched_gt:
                    continue
                matched_pred.add(pi)
                matched_gt.add(gi)
                tp_pairs.append((pi, gi, iou_val))

        tp = len(tp_pairs)
        fp = len(preds) - tp
        fn = len(gts)   - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        iou_tp_list.extend([t[2] for t in tp_pairs])
        per_image.append({
            "image_id" : img_id,
            "n_gt"     : len(gts),
            "n_pred"   : len(preds),
            "TP"       : tp,
            "FP"       : fp,
            "FN"       : fn,
            "mean_iou" : (round(float(np.mean([t[2] for t in tp_pairs])), 4)
                          if tp_pairs else None),
        })

    precision = (total_tp / (total_tp + total_fp)
                 if (total_tp + total_fp) > 0 else 0.0)
    recall    = (total_tp / (total_tp + total_fn)
                 if (total_tp + total_fn) > 0 else 0.0)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    mean_iou  = float(np.mean(iou_tp_list)) if iou_tp_list else 0.0

    return {
        "iou_threshold": iou_threshold,
        "tp"           : total_tp,
        "fp"           : total_fp,
        "fn"           : total_fn,
        "precision"    : round(precision, 4),
        "recall"       : round(recall,    4),
        "f1"           : round(f1,        4),
        "mean_iou_tp"  : round(mean_iou,  4),
        "per_image"    : per_image,
    }


def print_evaluation(metrics: dict):
    bar = "═" * 54
    sep = "─" * 54
    print(f"\n{bar}")
    print(f"  KẾT QUẢ ĐÁNH GIÁ  (IoU threshold = {metrics['iou_threshold']})")
    print(bar)
    print(f"  TP (True Positive)    : {metrics['tp']:>6}")
    print(f"  FP (False Positive)   : {metrics['fp']:>6}")
    print(f"  FN (False Negative)   : {metrics['fn']:>6}")
    print(sep)
    print(f"  Precision             : {metrics['precision']:>6.4f}  "
          f"({metrics['precision']*100:.2f}%)")
    print(f"  Recall                : {metrics['recall']:>6.4f}  "
          f"({metrics['recall']*100:.2f}%)")
    print(f"  F1 Score  ★           : {metrics['f1']:>6.4f}  "
          f"({metrics['f1']*100:.2f}%)")
    print(f"  Mean IoU (TP only)    : {metrics['mean_iou_tp']:>6.4f}")
    print(bar)

    bad = sorted(
        [r for r in metrics["per_image"] if r["FP"] + r["FN"] > 0],
        key=lambda r: r["FP"] + r["FN"], reverse=True,
    )[:10]
    if bad:
        print("\n  Top ảnh có nhiều lỗi nhất:")
        hdr = (f"  {'image_id':<44} {'GT':>4} {'Pred':>4} "
               f"{'TP':>4} {'FP':>4} {'FN':>4} {'IoU':>6}")
        print(hdr)
        print(f"  {'-'*44} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*6}")
        for r in bad:
            iou_s = f"{r['mean_iou']:.4f}" if r["mean_iou"] is not None else "  —   "
            print(f"  {r['image_id']:<44} {r['n_gt']:>4} {r['n_pred']:>4} "
                  f"{r['TP']:>4} {r['FP']:>4} {r['FN']:>4} {iou_s:>6}")
    print()


def save_evaluation_csv(metrics: dict, out_path: str):
    rows = metrics["per_image"]
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[*] Chi tiết per-image → {out_path}")


FIELDNAMES = [
    "image_id", "qr_index",
    "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3",
    "content",
]


def write_output(rows: list, output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[*] Đã lưu kết quả → {output_path}  ({len(rows)} hàng)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="QR Code OBB Detection + Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data",      default=None,
                        help="CSV đầu vào (image_id, image_path).")
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    parser.add_argument("--output",    default=DEFAULT_OUTPUT)
    parser.add_argument("--no-decode", action="store_true",
                        help="Bỏ decode nội dung QR.")
    parser.add_argument("--conf",      type=float, default=CONF_THRESHOLD)
    parser.add_argument("--device",    default=DEVICE,
                        help='"cpu" | "0" | "cuda:0"')
    parser.add_argument("--gt",        default=None,
                        help="CSV ground-truth để đánh giá sau inference.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Chỉ đánh giá (không inference). Cần --pred + --gt.")
    parser.add_argument("--pred",      default=None,
                        help="CSV dự đoán sẵn (dùng với --eval-only).")
    parser.add_argument("--iou-thr",   type=float, default=0.5,
                        help="Ngưỡng IoU matching (mặc định 0.5).")
    parser.add_argument("--eval-csv",  default=None,
                        help="Lưu kết quả đánh giá per-image ra file này.")
    return parser.parse_args()


def run_inference(args) -> str:
    global CONF_THRESHOLD, DEVICE
    CONF_THRESHOLD = args.conf
    DEVICE         = args.device

    csv_dir = Path(args.data).parent
    df      = pd.read_csv(args.data)
    missing = {"image_id", "image_path"} - set(df.columns)
    if missing:
        raise ValueError(f"File CSV thiếu cột: {missing}")

    print(f"[*] Đầu vào : {args.data}  ({len(df)} ảnh)")
    print(f"[*] Model   : {args.model}")
    print(f"[*] Device  : {DEVICE}  |  Conf: {CONF_THRESHOLD}")
    print(f"[*] Output  : {args.output}")

    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
    except ImportError:
        raise ImportError("Cần cài ultralytics: pip install ultralytics")

    output_rows = []
    t_start     = time.time()

    for i, row in df.iterrows():
        image_id       = str(row["image_id"])
        image_path     = str(row["image_path"])
        image_path_abs = (image_path if os.path.isabs(image_path)
                          else str(csv_dir / image_path))

        if not os.path.exists(image_path_abs):
            print(f"  [!] Không tìm thấy: {image_path_abs}")
            output_rows.append({
                "image_id": image_id, "qr_index": "",
                **{c: "" for c in COORD_COLS}, "content": "",
            })
            continue

        detections = detect_qr_in_image(
            model, image_path_abs, decode=not args.no_decode)

        if not detections:
            output_rows.append({
                "image_id": image_id, "qr_index": "",
                **{c: "" for c in COORD_COLS}, "content": "",
            })
        else:
            for det in detections:
                output_rows.append({"image_id": image_id, **det})

        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            elapsed = time.time() - t_start
            n_qr    = sum(1 for r in output_rows if r["qr_index"] != "")
            print(f"  [{i+1:>5}/{len(df)}]  "
                  f"{elapsed/(i+1):.3f}s/ảnh  QRs: {n_qr}")

    elapsed = time.time() - t_start
    print(f"\n[*] Tổng thời gian: {elapsed:.1f}s  "
          f"({elapsed/len(df):.3f}s/ảnh trung bình)")

    write_output(output_rows, args.output)

    n_det     = sum(1 for r in output_rows if r["qr_index"] != "")
    n_no_qr   = sum(1 for r in output_rows if r["qr_index"] == "")
    n_decoded = sum(1 for r in output_rows if r.get("content", ""))
    print(f"[*] Có QR: {n_det}  |  Không QR: {n_no_qr}"
          f"  |  Đọc được nội dung: {n_decoded}")

    return args.output


def main():
    args = parse_args()

    if args.eval_only:
        if not args.pred or not args.gt:
            raise ValueError("--eval-only cần cả --pred và --gt.")
        print(f"[*] Đánh giá: pred={args.pred}  gt={args.gt}")
        metrics = evaluate(args.pred, args.gt, iou_threshold=args.iou_thr)
        print_evaluation(metrics)
        if args.eval_csv:
            save_evaluation_csv(metrics, args.eval_csv)
        return

    if not args.data:
        raise ValueError("Cần truyền --data (hoặc dùng --eval-only).")

    pred_path = run_inference(args)

    if args.gt:
        print(f"\n[*] So sánh với ground-truth: {args.gt}")
        metrics = evaluate(pred_path, args.gt, iou_threshold=args.iou_thr)
        print_evaluation(metrics)
        if args.eval_csv:
            save_evaluation_csv(metrics, args.eval_csv)


if __name__ == "__main__":
    main()