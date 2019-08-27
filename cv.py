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
    cv2.circle(img, (int(x), int(y)), int(radius), color, thick)


def cv_point(img, center, color, r=1):
    cv_circle(img, center, color, r, -1)


def cv_size(img):
    return img.shape


def cv_scatter(img, xs, ys, xr=None, yr=None, c=None, r=None):
    size = cv_size(img)
    if not c:
        c = [(128, 128, 128)] * len(xs)
    if not r:
        r = [int((size[0] + size[1]) / 800)] * len(xs)
    if not xr:
        xr = [min(xs), max(xs)]
    if not yr:
        yr = [min(ys), max(ys)]
    trans_x: Callable[[Any], Any] = lambda v: (v - xr[0]) / (xr[1] - xr[0]) * (size[0] - 1)
    trans_y: Callable[[Any], Any] = lambda v: (v - yr[0]) / (yr[1] - yr[0]) * (size[1] - 1)
    for [x, y, color, radius] in zip(xs, ys, c, r):
        cv_point(img, (trans_x(x), trans_y(y)), color, radius)
    # cv_show(img)
    return img


def trans_fn(from_min, from_max, to_min, to_max):
    return lambda x: (x - from_min) / (from_max - from_min) * (to_max - to_min - 0.00000001) + to_min


def hsv2rgb(h, s, v):
    import math
    import colorsys
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v


def main():
    pass


if __name__ == '__main__':
    main()
