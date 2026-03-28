import cv2
import numpy as np


def read_img(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def stitch(img1, img2, kp1, kp2, matches):
    # img2 -> img1 로 붙이기 위해 (pts2 -> pts1)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)  # ★ 방향 반대로
    if H is None:
        return None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # img1을 기준 캔버스의 왼쪽에 두고, img2를 워핑해서 오른쪽으로 이어붙임
    canvas = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))  # ★ img2를 warp
    canvas[0:h1, 0:w1] = img1  # ★ 기준 이미지는 img1
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    canvas = canvas[y:y+h, x:x+w]

    return canvas


def main():
    img1 = read_img("IMG_6263.jpg")
    img2 = read_img("IMG_6264.jpg")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 오매칭 줄이기 (과제 범위 내에서 가장 간단한 안정화)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)  # 좋은 매칭부터

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    cv2.imwrite("sift_match.png", match_img)

    result = stitch(img1, img2, kp1, kp2, matches)
    if result is None:
        print("Homography failed (매칭이 부족하거나 오매칭이 많음)")
        return

    cv2.imwrite("sift_stitch.png", result)
    print("SIFT done")


if __name__ == "__main__":
    main()
