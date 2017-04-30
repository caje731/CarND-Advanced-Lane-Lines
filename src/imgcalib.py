"""Camera Calibration"""
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

class CameraCalibrator:
    """A class that provides image calibration and undistortion."""

    def __init__(self):
        self._cam_matrix = None
        self._dist_coeffs = None
        self._is_calib = False
        self._objpoints = None
        self._imgpoints = None

    def load_calibration_points(self, images_path, visualize=False):
        """prepare object points and image points from all the images"""

        if len(images_path) == 0:
            raise ValueError('No images path provided')

        images = glob.glob(images_path)

        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

                if visualize:
                    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.set_title('Original Image {}'.format(img.shape))
                    ax1.imshow(img)

                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                    ax2.set_title('Chessboard corners {}'.format(img.shape))
                    ax2.imshow(img)
                    plt.show()
            else:
                if visualize:
                    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.set_title('Original Image {}'.format(img.shape))
                    ax1.imshow(img)

                    ax2.set_title('Chessboard not found')
                    plt.show()

        self._objpoints = objpoints
        self._imgpoints = imgpoints

    def calibrate_camera(self, src_img):
        """Calculate camera matrix and distortion coeffs given a source image. """

        if self._objpoints is None or self._imgpoints is None:
            raise ValueError('Image points do not exist')

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._objpoints,
            self._imgpoints,
            (src_img.shape[1], src_img.shape[0]), # (w, h) for internal cam mtx
            None,   # Cam mtx
            None    # distortion coeffs
        )
        self._cam_matrix = mtx
        self._dist_coeffs = dist
        self._is_calib = True

    def undistort(self, distorted_img):
        """Transform an image to compensate for lens distortion."""

        if not self._is_calib:
            self.calibrate_camera(distorted_img)

        return cv2.undistort(
            distorted_img,
            self._cam_matrix,
            self._dist_coeffs,
            None,               # pointer to output image
            self._cam_matrix    # new matrix - for additional scaling/shifting
        )
