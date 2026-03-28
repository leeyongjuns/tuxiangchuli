# visualize_image.py
import os
import argparse
import numpy as np
import cv2
import joblib

import Sliding as sd
from skimage.feature import hog
from skimage.transform import pyramid_gaussian

from nms import nms


def detect_one_image(
    img_path: str,
    model_path: str = "models/models.dat",
    out_dir: str = "output",
    resize_w: int = 400,
    win_size=(64, 128),
    step_size=(9, 9),
    downscale: float = 1.25,
    score_thresh: float = 0.6,
    iou_thresh: float = 0.3,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) load model
    model = joblib.load(model_path)

    # 2) load image
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"이미지 로드 실패: {img_path}")

    # keep aspect ratio resize (너 원본 코드 스타일 따라감)
    h, w = image.shape[:2]
    new_w = resize_w
    new_h = int(h * (new_w / float(w)))
    image = cv2.resize(image, (new_w, new_h))
    orig = image.copy()

    rects = []
    scores = []

    # 3) pyramid
    for resized in pyramid_gaussian(image, downscale=downscale, channel_axis=-1):
        # pyramid_gaussian returns float [0,1] sometimes → uint8 변환
        if resized.dtype != np.uint8:
            resized = (resized * 255).astype(np.uint8)

        if resized.shape[0] < win_size[1] or resized.shape[1] < win_size[0]:
            break

        scale_x = orig.shape[1] / float(resized.shape[1])
        scale_y = orig.shape[0] / float(resized.shape[0])

        # 4) sliding window (IMPORTANT: 네 Sliding.py 시그니처 그대로 사용)
        for (x, y, window) in sd.sliding_window(resized, win_size, step_size):
            if window.shape[0] != win_size[1] or window.shape[1] != win_size[0]:
                continue

            gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

            # 5) HOG feature (training_SVM.py와 동일 파라미터)
            fd = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(3, 3),
                visualize=False,
            )
            fd = fd.reshape(1, -1).astype(np.float32)

            # 6) SVM score
            sc = float(model.decision_function(fd)[0])
            if sc > score_thresh:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + win_size[0]) * scale_x)
                y2 = int((y + win_size[1]) * scale_y)
                rects.append([x1, y1, x2, y2])
                scores.append(sc)

    rects = np.array(rects, dtype=np.int32) if len(rects) else np.zeros((0, 4), dtype=np.int32)
    scores = np.array(scores, dtype=np.float32) if len(scores) else np.zeros((0,), dtype=np.float32)

    # 7) NMS
    keep = nms(rects, scores, iou_thresh=iou_thresh)

    # 8) draw
    vis = orig.copy()
    for i in keep:
        x1, y1, x2, y2 = rects[i]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"person {scores[i]:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"detect_{base}.jpg")
    cv2.imwrite(out_path, vis)

    print(f"dets(before nms)={len(rects)}, after nms={len(keep)}")
    print("saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="e.g. test/12.jpg")
    ap.add_argument("--model", default="models/models.dat")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--score", type=float, default=0.6)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--step", type=int, default=9)
    ap.add_argument("--downscale", type=float, default=1.25)
    args = ap.parse_args()

    detect_one_image(
        img_path=args.img,
        model_path=args.model,
        out_dir=args.outdir,
        step_size=(args.step, args.step),
        downscale=args.downscale,
        score_thresh=args.score,
        iou_thresh=args.iou,
    )


if __name__ == "__main__":
    main()
