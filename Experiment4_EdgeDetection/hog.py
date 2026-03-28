# -*- coding: utf-8 -*-
"""
HOG visualization script (ROI 선택 -> 64x128 -> HOG 시각화 저장)
- 사용법:
  python3 hog.py --img test/me2.jpg
  python3 hog.py --img test/12.jpg
  python3 hog.py --img test/다른사진.jpg
- 결과:
  output/hog_vis_<파일명>.png
  output/hog_input_<파일명>.png  (64x128 입력 패치)
"""

import os
import cv2
import numpy as np
import math
import argparse
import matplotlib
matplotlib.use("Agg")  # 창 띄우지 않음
import matplotlib.pyplot as plt


class Hog_descriptor:
    def __init__(self, img, cell_size=8, bin_size=9):
        self.img = img

        # gamma correction (0.5)
        m = float(np.max(img)) if np.max(img) > 0 else 1.0
        self.img = np.sqrt(img * 1.0 / m) * 255.0

        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 180 / self.bin_size
        assert type(self.bin_size) == int
        assert type(self.cell_size) == int
        assert 180 % self.bin_size == 0

    def extract(self):
        height, width = self.img.shape

        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)

        cell_gradient_vector = np.zeros(
            (int(height / self.cell_size), int(width / self.cell_size), self.bin_size),
            dtype=np.float64
        )

        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[
                    i * self.cell_size:(i + 1) * self.cell_size,
                    j * self.cell_size:(j + 1) * self.cell_size
                ]
                cell_angle = gradient_angle[
                    i * self.cell_size:(i + 1) * self.cell_size,
                    j * self.cell_size:(j + 1) * self.cell_size
                ]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width], dtype=np.float64), cell_gradient_vector)

        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])

                mag = lambda v: math.sqrt(sum(x ** 2 for x in v))
                magnitude = mag(block_vector) + 1e-5
                if magnitude != 0:
                    block_vector = [x / magnitude for x in block_vector]
                hog_vector.append(block_vector)

        return np.asarray(hog_vector), hog_image

    def global_gradient(self):
        gx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        ang[ang > 180.0] -= 180.0
        return mag, ang

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0.0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                strength = cell_magnitude[i][j]
                angle = cell_angle[i][j]
                min_bin, max_bin, weight = self.get_closest_bins(angle)
                orientation_centers[min_bin] += (strength * (1 - weight))
                orientation_centers[max_bin] += (strength * weight)
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx % self.bin_size, (idx + 1) % self.bin_size, mod / self.angle_unit

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        if max_mag <= 0:
            return image

        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y].copy()
                cell_grad /= max_mag

                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)

                    x1 = int(x * self.cell_size + cell_width + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + cell_width + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size + cell_width - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size + cell_width - magnitude * cell_width * math.sin(angle_radian))

                    cv2.line(image, (y1, x1), (y2, x2), float(255.0 * math.sqrt(max(magnitude, 0.0))))
                    angle += angle_gap
        return image


def safe_basename(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    # 파일명에 한글/특수문자 있어도 저장은 되는데, 안전하게 영문/숫자/언더스코어만 남기고 싶으면 아래를 사용
    # 여기선 원본 name을 그대로 쓰되, 공백만 언더스코어로
    return name.replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="input image path (e.g., test/me2.jpg)")
    ap.add_argument("--outdir", default="output", help="output dir")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지 로드 실패: {args.img}")

    # ROI 선택 (OpenCV 창에서 드래그 후 ENTER/SPACE)
    print("Select ROI (drag mouse, press ENTER)")
    roi = cv2.selectROI("Select ROI (drag, then Enter/Space)", img_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("ROI 선택이 취소됨(또는 크기 0). 다시 실행해서 ROI를 드래그하고 Enter/Space 누르기")

    roi_bgr = img_bgr[int(y):int(y+h), int(x):int(x+w)]
    roi_bgr = cv2.resize(roi_bgr, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

    # matplotlib 입력용 RGB
    roi_rgb = roi_bgr[:, :, ::-1]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    hog = Hog_descriptor(gray, cell_size=8, bin_size=9)
    hog_vector, hog_image = hog.extract()

    print("hog_vector", hog_vector.shape)
    print("hog_image", hog_image.shape)

    tag = safe_basename(args.img)

    # 1) 64x128 입력 저장
    input_path = os.path.join(args.outdir, f"hog_input_{tag}.png")
    cv2.imwrite(input_path, roi_bgr)

    # 2) 시각화 저장 (창 없이 파일로)
    vis_path = os.path.join(args.outdir, f"hog_vis_{tag}.png")
    plt.figure(figsize=(6.4, 6.4))
    plt.subplot(1, 2, 1)
    plt.title("Input (64x128)")
    plt.imshow(roi_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("HOG visualization")
    plt.imshow(hog_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(vis_path, dpi=200)
    plt.close()

    print("saved:", vis_path)
    print("saved:", input_path)


if __name__ == "__main__":
    main()
