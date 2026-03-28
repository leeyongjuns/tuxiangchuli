import os
import math
import random
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


def mkdir(p):
    os.makedirs(p, exist_ok=True)


def read_img(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cannot read image: " + p)
    return img


def write_img(p, img):
    d = os.path.dirname(p)
    if d:
        mkdir(d)
    cv2.imwrite(p, img)


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def l2_loss(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.sum((a - b) ** 2))


def ssim_simple(g1, g2):
    x = g1.astype(np.float64)
    y = g2.astype(np.float64)

    mu_x = x.mean()
    mu_y = y.mean()

    var_x = x.var()
    var_y = y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()

    L = 255.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if den == 0:
        return 0.0
    return float(num / den)


def list_images(d):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    out = []
    for fn in os.listdir(d):
        if fn.lower().endswith(exts):
            out.append(os.path.join(d, fn))
    out.sort()
    return out


def damage_line(img, seed=0, width=1):
    rnd = random.Random(seed)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for _ in range(3):
        x1 = rnd.randint(0, w - 1)
        y1 = rnd.randint(0, h - 1)
        ang = rnd.random() * 2 * math.pi
        length = rnd.randint(int(0.5 * min(h, w)), int(0.95 * min(h, w)))
        x2 = int(x1 + length * math.cos(ang))
        y2 = int(y1 + length * math.sin(ang))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=width)

    damaged = img.copy()
    damaged[mask == 1] = 255
    return damaged, mask


def damage_drop(img, ratio, seed=0):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    n = h * w
    k = int(n * ratio)
    idx = rng.choice(n, size=k, replace=False)

    mask = np.zeros(n, dtype=np.uint8)
    mask[idx] = 1
    mask = mask.reshape(h, w)

    damaged = img.copy()
    damaged[mask == 1] = 0
    return damaged, mask


def restore_nearest(damaged, mask):
    h, w = mask.shape
    _, labels = cv2.distanceTransformWithLabels(
        mask.astype(np.uint8),
        distanceType=cv2.DIST_L2,
        maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL
    )

    out = damaged.copy()
    miss = np.argwhere(mask == 1)
    for i in range(len(miss)):
        y = int(miss[i][0])
        x = int(miss[i][1])
        lab = int(labels[y, x])
        if lab <= 0:
            continue
        idx = lab - 1
        yy = idx // w
        xx = idx % w
        out[y, x] = damaged[yy, xx]
    return out


def interp_1d(v, known):
    n = v.shape[0]
    x = np.arange(n)
    kp = x[known]
    if kp.size == 0:
        return v
    vp = v[known]
    out = v.copy()
    out[~known] = np.interp(x[~known], kp, vp)
    return out


def restore_bilinear(damaged, mask):
    h, w = mask.shape
    known = (mask == 0)

    a = damaged.astype(np.float64)
    row = a.copy()
    col = a.copy()

    for y in range(h):
        k = known[y, :]
        if np.any(k):
            row[y, :, 0] = interp_1d(row[y, :, 0], k)
            row[y, :, 1] = interp_1d(row[y, :, 1], k)
            row[y, :, 2] = interp_1d(row[y, :, 2], k)

    for x in range(w):
        k = known[:, x]
        if np.any(k):
            col[:, x, 0] = interp_1d(col[:, x, 0], k)
            col[:, x, 1] = interp_1d(col[:, x, 1], k)
            col[:, x, 2] = interp_1d(col[:, x, 2], k)

    out = a.copy()
    miss = (mask == 1)
    out[miss] = 0.5 * row[miss] + 0.5 * col[miss]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def rbf_phi(r, eps):
    return np.exp(-(eps * r) ** 2)


def restore_rbf(damaged, mask, window=21, kmax=30, eps=0.08, lam=1e-3):
    h, w = mask.shape
    half = window // 2

    a = damaged.astype(np.float64)
    out = a.copy()

    known = (mask == 0)
    pts = np.argwhere(known)
    if pts.size == 0:
        return damaged.copy()

    ky = pts[:, 0]
    kx = pts[:, 1]
    kv = out[ky, kx, :]

    miss = np.argwhere(mask == 1)
    for i in range(len(miss)):
        y = int(miss[i][0])
        x = int(miss[i][1])

        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)
        x0 = max(0, x - half)
        x1 = min(w, x + half + 1)

        ok = (ky >= y0) & (ky < y1) & (kx >= x0) & (kx < x1)
        idxs = np.where(ok)[0]

        if idxs.size < 4:
            dy = ky - y
            dx = kx - x
            j = int(np.argmin(dy * dy + dx * dx))
            out[y, x, :] = kv[j, :]
            continue

        dy = ky[idxs] - y
        dx = kx[idxs] - x
        d2 = dy * dy + dx * dx
        order = np.argsort(d2)[:min(kmax, idxs.size)]
        sel = idxs[order]

        p = np.stack([ky[sel], kx[sel]], axis=1).astype(np.float64)
        v = kv[sel, :]
        K = p.shape[0]

        diff = p[:, None, :] - p[None, :, :]
        rij = np.sqrt(np.sum(diff ** 2, axis=2))
        Phi = rbf_phi(rij, eps) + lam * np.eye(K)

        r0 = np.sqrt(np.sum((p - np.array([y, x], dtype=np.float64)) ** 2, axis=1))
        phi0 = rbf_phi(r0, eps)

        pred = np.zeros(3, dtype=np.float64)
        try:
            for c in range(3):
                wgt = np.linalg.solve(Phi, v[:, c])
                pred[c] = float(np.dot(wgt, phi0))
            out[y, x, :] = pred
        except Exception:
            out[y, x, :] = v[0, :]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def run_task1(img_paths, out_dir, seed=0):
    mkdir(out_dir)
    for i, p in enumerate(img_paths):
        img = read_img(p)

        damaged, mask = damage_line(img, seed=seed + i, width=1)

        nn = restore_nearest(damaged, mask)
        bl = restore_bilinear(damaged, mask)
        rb = restore_rbf(damaged, mask)

        name = os.path.splitext(os.path.basename(p))[0]
        write_img(os.path.join(out_dir, name + "_original.png"), img)
        write_img(os.path.join(out_dir, name + "_damaged.png"), damaged)
        write_img(os.path.join(out_dir, name + "_mask.png"), (mask * 255).astype(np.uint8))
        write_img(os.path.join(out_dir, name + "_nearest.png"), nn)
        write_img(os.path.join(out_dir, name + "_bilinear.png"), bl)
        write_img(os.path.join(out_dir, name + "_rbf.png"), rb)


def run_task2(img_path, out_dir, seed=0, save_ratio=0.1):
    mkdir(out_dir)
    print("task2 start:", img_path)

    img = read_img(img_path)

    if max(img.shape[0], img.shape[1]) > 300:
        s = 600 / max(img.shape[0], img.shape[1])
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))

    ratios = [i / 100.0 for i in range(10, 100, 10)]
    xs = [int(r * 100) for r in ratios]

    l2_nn, l2_bl, l2_rb = [], [], []
    s_nn, s_bl, s_rb = [], [], []

    g0 = to_gray(img)

    for i, r in enumerate(ratios):
        damaged, mask = damage_drop(img, r, seed=seed + i)

        nn = restore_nearest(damaged, mask)
        bl = restore_bilinear(damaged, mask)
        rb = restore_rbf(damaged, mask, window=9, kmax=8, eps=0.15, lam=1e-3)

        l2_nn.append(l2_loss(img, nn))
        l2_bl.append(l2_loss(img, bl))
        l2_rb.append(l2_loss(img, rb))

        s_nn.append(ssim_simple(g0, to_gray(nn)))
        s_bl.append(ssim_simple(g0, to_gray(bl)))
        s_rb.append(ssim_simple(g0, to_gray(rb)))

        if abs(r - save_ratio) < 1e-12:
            name = os.path.splitext(os.path.basename(img_path))[0]
            rr = int(r * 100)
            write_img(os.path.join(out_dir, f"{name}_ratio{rr}_damaged.png"), damaged)
            write_img(os.path.join(out_dir, f"{name}_ratio{rr}_mask.png"), (mask * 255).astype(np.uint8))
            write_img(os.path.join(out_dir, f"{name}_ratio{rr}_nearest.png"), nn)
            write_img(os.path.join(out_dir, f"{name}_ratio{rr}_bilinear.png"), bl)
            write_img(os.path.join(out_dir, f"{name}_ratio{rr}_rbf.png"), rb)

    print("task2 save graphs...")
    print("ratio", int(r*100), "%")


    plt.figure()
    plt.plot(xs, l2_nn, marker="o", label="Nearest Neighbor")
    plt.plot(xs, l2_bl, marker="x", linestyle="--", label="Bilinear")
    plt.plot(xs, l2_rb, marker="s", linestyle="-.", label="RBF")
    plt.xlabel("Perturbation Ratio (%)")
    plt.ylabel("L2 Loss")
    plt.title("L2 Loss Comparison (Random Pixel Drop)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "l2loss_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(xs, s_nn, marker="o", label="Nearest Neighbor")
    plt.plot(xs, s_bl, marker="x", linestyle="--", label="Bilinear")
    plt.plot(xs, s_rb, marker="s", linestyle="-.", label="RBF")
    plt.xlabel("Perturbation Ratio (%)")
    plt.ylabel("SSIM")
    plt.title("SSIM Comparison (Random Pixel Drop)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ssim_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close()

    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("ratio(%)\tNN_L2\tBL_L2\tRBF_L2\tNN_SSIM\tBL_SSIM\tRBF_SSIM\n")
        for i in range(len(xs)):
            f.write(
                f"{xs[i]}\t"
                f"{l2_nn[i]:.6e}\t{l2_bl[i]:.6e}\t{l2_rb[i]:.6e}\t"
                f"{s_nn[i]:.6f}\t{s_bl[i]:.6f}\t{s_rb[i]:.6f}\n"
            )

    print("task2 done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_ratio", type=int, default=10)
    ap.add_argument("--task2_img", default="")
    args = ap.parse_args()

    imgs = list_images(args.images_dir)
    print("images:", len(imgs))

    if len(imgs) < 1:
        raise RuntimeError("no images found in: " + args.images_dir)

    mkdir(args.out_dir)

    run_task1(imgs, os.path.join(args.out_dir, "task1"), seed=args.seed)

    if args.task2_img.strip():
        t2 = args.task2_img.strip()
    else:
        t2 = imgs[0]

    run_task2(
        t2,
        os.path.join(args.out_dir, "task2"),
        seed=args.seed + 100,
        save_ratio=args.save_ratio / 100.0
    )

    print("done. check:", args.out_dir)


if __name__ == "__main__":
    main()
