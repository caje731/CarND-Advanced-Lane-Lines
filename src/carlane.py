#!/usr/bin/python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# This class represents a car lane that has left and right car lines
class CarLane:

    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    def __init__(self):
        self.counter = 0 # Tick tick tick
        self.left = Line()
        self.right = Line()
        self.vehicle_position = 0.0
        self.radius_of_curvature = 0.0
        self.vehicle_position_text = '' # TODO: Move to a separate view model class
        self.radius_of_curvature_text = '' # TODO: Move to a separate view model class

    def get_vehicle_position(self, image_shape):
        """ Return the position of vehicle """
        self.vehicle_position = abs(image_shape[1] // 2 - ((self.left.line_base_pos + self.right.line_base_pos) / 2))
        if self.vehicle_position_text == '' or self.counter % 3 == 0:
            vpos_direction = 'left' if self.vehicle_position < image_shape[1] // 2 else 'right'
            self.vehicle_position_text = '{:.2f}m {} of center'.format(self.vehicle_position * CarLane.xm_per_pix, vpos_direction)
        return self.vehicle_position_text

    def get_radius_of_curvature(self):
        """ Return the radius of curvature """
        self.radius_of_curvature = (self.left.radius_of_curvature + self.right.radius_of_curvature) // 2
        if self.radius_of_curvature_text == '' or self.counter % 3 == 0:
            self.radius_of_curvature_text = self.radius_of_curvature
        return self.radius_of_curvature_text


    def find_lane_pixels(self, binary_warped):
        """ Initially, this method will route to the find_lane_pixels_ws.
            Then, it will route to the find_lane_pixels_fast to speed up the pixel findings.
         """
        self.counter += 1
        if self.left.line_fit is None or self.right.line_fit is None:
            return self.__find_lane_pixels_ws(binary_warped)
        else:
            return self.__find_lane_pixels_fast(binary_warped, self.left.line_fit, self.right.line_fit)

    def __find_lane_pixels_ws(self, binary_warped):
        """ Find lane line pixels using sliding windows. """

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint


        # Set height of windows
        nwindows = 9  # Choose the number of sliding windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # TODO: Recover
        self.left.detected = len(left_lane_inds) > 0
        self.right.detected = len(right_lane_inds) > 0

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def __find_lane_pixels_fast(self, binary_warped, left_fit, right_fit):
        """ Find lane line pixels efficiently by skipping the sliding windows step """

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 100

        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # TODO: Recover
        self.left.detected = len(left_lane_inds) > 0
        self.right.detected = len(right_lane_inds) > 0

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    """ Prototype methods """
    def visualize_polyfit_lines(self, binary_warped, out_img, left_fitx, right_fitx, ploty):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
#        ax1.imshow(binary_warped, cmap='gray')
        ax1.set_title('Binary warped', fontsize=20)

#        ax2.imshow(out_img)
        ax2.set_title('Polynomial fit', fontsize=20)
        ax2.plot(left_fitx, ploty, color='yellow')
        ax2.plot(right_fitx, ploty, color='yellow')
        ax2.set_xlim(0, 1280)
        ax2.set_ylim(720, 0)

    def polyfit_lines(self, binary_warped, visualize=False):

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        if visualize:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        nwindows = 9  # Choose the number of sliding windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            if visualize:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
                # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        leftx_int = left_fit[0] * binary_warped.shape[0] ** 2 + left_fit[1] * binary_warped.shape[0] + left_fit[2]
        rightx_int = right_fit[0] * binary_warped.shape[0] ** 2 + right_fit[1] * binary_warped.shape[0] + right_fit[2]

        if visualize:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        image_shape = binary_warped.shape

        y_eval = np.max(ploty)

        #     # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * CarLane.ym_per_pix, leftx * CarLane.xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * CarLane.ym_per_pix, rightx * CarLane.xm_per_pix, 2)
        #     # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * CarLane.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * CarLane.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        radius_of_curvature = int((left_curverad + right_curverad) / 2)

        center = abs(image_shape[1] // 2 - ((rightx_int + leftx_int) / 2))

        if visualize: self.visualize_polyfit_lines(binary_warped, out_img, left_fitx, right_fitx, ploty)

        return left_fitx, right_fitx, ploty, radius_of_curvature, center

    def overlay_image(self, warped, original_img, Minv, left_fitx, right_fitx, ploty):
        image = original_img

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        return cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

# Define a class to receive the characteristics of each line detection
class Line:

    MAX_QUEUE_LENGTH = 3

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        #self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        #self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        #self.best_fit = None
        #polynomial coefficients for the most recent fit
        #self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        self.line_fit = None # Store the line fit

        # Store recent polynomial coefficients for averaging across frames
        self.line_fit0_queue = deque(maxlen=Line.MAX_QUEUE_LENGTH)
        self.line_fit1_queue = deque(maxlen=Line.MAX_QUEUE_LENGTH)
        self.line_fit2_queue = deque(maxlen=Line.MAX_QUEUE_LENGTH)

        self.ploty = None

    def polyfit_lines(self, allx, ally, image_shape):
        """ Find the polynomial fit based on the discovered line points coordinates """

        #flatten = lambda l: [item for sublist in l for item in sublist]

        if len(allx) > 0 and len(ally) > 0:
            self.allx = allx
            self.ally = ally

            # Fit a second order polynomial to each
            self.line_fit = np.polyfit(ally, allx, 2)
            self.line_fit0_queue.append(self.line_fit[0])
            self.line_fit1_queue.append(self.line_fit[1])
            self.line_fit2_queue.append(self.line_fit[2])
        else: # Recover the missing x or y using the previous polyfit data
            print('Recover using previous points')
            #line_fit = [self.line_fit0_queue[0], self.line_fit1_queue[0], self.line_fit2_queue[0]]
            self.line_fit = [np.mean(self.line_fit0_queue), np.mean(self.line_fit1_queue), np.mean(self.line_fit2_queue)]

        if self.ploty is None:
            self.ploty = np.linspace(0, image_shape[0] - 1, image_shape[0])
        line_fitx = self.line_fit[0] * self.ploty ** 2 + self.line_fit[1] * self.ploty + self.line_fit[2]
        line_fitx_int = self.line_fit[0] * image_shape[0] ** 2 + self.line_fit[1] * image_shape[0] + self.line_fit[2]

        y_eval = np.max(self.ploty)

        # Fit new polynomials to x,y in world space
        line_fit_cr = np.polyfit(self.ally * CarLane.ym_per_pix, self.allx * CarLane.xm_per_pix, 2)
        #     # Calculate the new radii of curvature
        line_curverad = ((1 + (2 * line_fit_cr[0] * y_eval * CarLane.ym_per_pix + line_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * line_fit_cr[0])
        self.radius_of_curvature = line_curverad
        self.line_base_pos = line_fitx_int

        return line_fitx
