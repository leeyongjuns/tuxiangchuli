import cv2
import numpy as np


def read_img(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def harris_detect(img, max_pts=500):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris = cv2.dilate(harris, None)

    thresh = 0.01 * harris.max()
    ys, xs = np.where(harris > thresh)

    keypoints = [cv2.KeyPoint(float(x), float(y), 3) for x, y in zip(xs, ys)]
    return keypoints[:max_pts]


def compute_descriptor(img, kps):
    sift = cv2.SIFT_create()
    kps, des = sift.compute(img, kps)
    return kps, des


def stitch(img1, img2, kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
    if H is None:
        return None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
    canvas[0:h1, 0:w1] = img1

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    canvas = canvas[y:y+h, x:x+w]

    return canvas


def main():
    img1 = read_img("IMG_6263.jpg")
    img2 = read_img("IMG_6264.jpg")

    kp1 = harris_detect(img1)
    kp2 = harris_detect(img2)

    kp1, des1 = compute_descriptor(img1, kp1)
    kp2, des2 = compute_descriptor(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    cv2.imwrite("harris_match.png", match_img)

    result = stitch(img1, img2, kp1, kp2, matches)
    if result is None:
        print("Harris homography failed")
        return

    cv2.imwrite("harris_stitch.png", result)
    print("Harris done")


if __name__ == "__main__":
    main()
