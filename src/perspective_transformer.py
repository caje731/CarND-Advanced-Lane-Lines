#!/usr/bin/python

import cv2
import numpy as np


class PerspectiveTransformer:
    def __init__(self):
        self.src = None
        self.dst = None
        pass

    # Warp image to provide a bird's eye view
    def transform_matrix(self, src, dst):
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def warp_image(self, undisorted_img, M):
        img_size = (undisorted_img.shape[1], undisorted_img.shape[0])
        warped = cv2.warpPerspective(undisorted_img, M, img_size)
        return warped

