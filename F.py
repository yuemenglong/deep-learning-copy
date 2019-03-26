import numpy as np
import cv2


def cv_sharp(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    img = cv2.filter2D(img, -1, kernel=kernel)
    return img
