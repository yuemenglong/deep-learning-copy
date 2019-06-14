import traceback
from typing import Any, Callable

import numpy as np

import cv2


def poly_area(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# cv sec

def cv_to_grey(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def cv_resize(img, size):
    return cv2.resize(img, size)


def cv_new(size):
    w, h = size
    return np.ones((w, h, 3), np.uint8) * 255


def cv_show(img, name="show"):
    if isinstance(img, str):
        img = cv2.imread(img)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv_load(path):
    return cv2.imread(path)


def cv_save(img, path):
    cv2.imwrite(path, img)


def cv_grid(imgs, size):
    (width, height) = size
    if len(imgs.shape) == 4:
        # 单通道
        rows = len(imgs)
        cols = len(imgs[0])
        total_width = width * cols
        total_height = height * rows
        grid = np.zeros((total_height, total_width))
        for r in range(rows):
            for c in range(cols):
                grid[r * height:(r + 1) * height, c * width:(c + 1) * width] = cv_resize(imgs[r][c], size)
        cv_show(grid)


def cv_line(img, p1, p2, color, thick=1):
    (l, t), (r, b) = p1, p2
    cv2.line(img, (int(l), int(t)), (int(r), int(b)), color, thick)


def cv_rect(img, p1, p2, color, thick=1):
    (l, t), (r, b) = p1, p2
    cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), color, thick)


def cv_circle(img, center, color, radius, thick=1):
    (x, y) = center
    cv2.circle(img, (int(x), int(y)), radius, color, thick)


def cv_point(img, center, color, r=1):
    cv_circle(img, center, color, r, -1)


def cv_size(img):
    return img.shape


def cv_scatter(img, xs, ys, xr=None, yr=None, c=None, r=None):
    size = cv_size(img)
    if not c:
        c = [(128, 128, 128)] * len(xs)
    if not r:
        r = [(size[0] + size[1]) / 800] * len(xs)
    if not xr:
        xr = [min(xs), max(xs)]
    if not yr:
        yr = [min(ys), max(ys)]
    trans_x: Callable[[Any], Any] = lambda v: (v - xr[0]) / (xr[1] - xr[0]) * size[0]
    trans_y: Callable[[Any], Any] = lambda v: (v - yr[0]) / (yr[1] - yr[0]) * size[1]
    for [x, y, color, radius] in zip(xs, ys, c, r):
        cv_point(img, (trans_x(x), trans_y(y)), color, radius)
    # cv_show(img)
    return img


def main():
    pass


if __name__ == '__main__':
    main()
