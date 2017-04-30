#!/usr/bin/python3
"""Advanced Lane Finding"""

import glob
import sys

from ipywidgets import fixed
from matplotlib.lines import Line2D
from moviepy.editor import VideoFileClip
import cv2
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np

from carlane import CarLane
from imageutil import ImageUtil
from imgcalib import CameraCalibrator
from perspective_transformer import PerspectiveTransformer
from videoutil import VideoUtil

CALIBRATION_PATH = 'camera_cal/calibration*.jpg'
TEST_IMAGES_PATH = 'test_images/test*.jpg'

_lane = CarLane()
_camera_calibrator = CameraCalibrator()
_perspective_transformer = PerspectiveTransformer()

# Load chessboard images and calculate the camera calibration matrix
_camera_calibrator.load_calibration_points(
    CALIBRATION_PATH,
    visualize=False#True
)
""" Verification of the distortion correction with the chessboard images
"""
for fname in glob.glob(CALIBRATION_PATH):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    undistorted_img = _camera_calibrator.undistort(gray)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Original Image {}'.format(gray.shape))
#    ax1.imshow(gray, cmap='gray')
    ax2.set_title('Undistorted Image {}'.format(undistorted_img.shape))
#    ax2.imshow(undistorted_img, cmap='gray')
    # plt.show()

""" Verification of the distortion correction with the test images """
for fname in glob.glob(TEST_IMAGES_PATH):
    original_img = cv2.imread(fname)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    undistorted_img = _camera_calibrator.undistort(original_img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Original Image {}'.format(original_img.shape))
#    ax1.imshow(original_img)
    ax2.set_title('Undistorted Image {}'.format(undistorted_img.shape))
#    ax2.imshow(undistorted_img)
    # plt.show()

""" Source and destination points """
src = np.float32([[585, 460],[203, 720], [1127, 720],[695, 460]])
dst = np.float32([[320, 0], [320, 720], [960, 720],[960, 0]])

""" Load transform matrix and inverse transform matrix """
M, Minv = _perspective_transformer.transform_matrix(src, dst)

""" Warp image to provide a bird's eye view """
(line1_xs, line1_ys) = zip(src[0],src[1])
(line2_xs, line2_ys) = zip(src[2], src[3])
(line3_xs, line3_ys) = zip(src[0], src[3])
(line4_xs, line4_ys) = zip(dst[0], dst[1])
(line5_xs, line5_ys) = zip(dst[2], dst[3])
(line6_xs, line6_ys) = zip(dst[0], dst[3])

for image_path in glob.glob(TEST_IMAGES_PATH):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undistorted_img = _camera_calibrator.undistort(image)
    color_warped = _perspective_transformer.warp_image(undistorted_img, M)
    f, (ax1, ax2) =plt.subplots(1, 2, figsize=(8,4))
    f.tight_layout()
#    ax1.imshow(undistorted_img)
    ax1.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='red',ls=':'))
    ax1.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color='red', ls=':'))
    ax1.add_line(Line2D(line3_xs, line3_ys, linewidth=1, color='red', ls=':'))
    ax1.set_title('Undistorted Image', fontsize=20)
#    ax2.imshow(color_warped)
    ax2.add_line(Line2D(line4_xs, line4_ys,linewidth=1, color='red',ls=':'))
    ax2.add_line(Line2D(line5_xs, line5_ys, linewidth=1, color='red', ls=':'))
    ax2.add_line(Line2D(line6_xs, line6_ys, linewidth=1, color='red', ls=':'))
    ax2.set_title('Color Warped Image', fontsize=20)

def explorer_perspective_transform_points(src0x=585, src0y=460, \
                                        src1x=203, src1y=720,   \
                                        src2x=1127, src2y=720,  \
                                        src3x=695, src3y=460,   \
                                        dst0x=320, dst0y=0,     \
                                        dst1x=320, dst1y=720,   \
                                        dst2x=960, dst2y=720,   \
                                        dst3x=960, dst3y=0):

    (line1_xs, line1_ys) = zip([src0x,src0y],[src1x,src1y])
    (line2_xs, line2_ys) = zip([src2x,src2y],[src3x,src3y])
    (line3_xs, line3_ys) = zip([src0x,src0y],[src3x,src3y])
    (line4_xs, line4_ys) = zip([dst0x,dst0y],[dst1x,dst1y])
    (line5_xs, line5_ys) = zip([dst2x,dst2y],[dst3x,dst3y])
    (line6_xs, line6_ys) = zip([dst0x, dst0y], [dst3x, dst3y])

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    f.tight_layout()
#    ax1.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    ax1.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='red', ls=':'))
    ax1.add_line(Line2D(line2_xs, line2_ys,linewidth=1,color='red',ls=':'))
    ax1.add_line(Line2D(line3_xs, line3_ys, linewidth=1, color='red',ls=':'))
    ax1.set_title('Undistorted Image',fontsize=20)
#    ax2.imshow(cv2.cvtColor(color_warped, cv2.COLOR_BGR2RGB))
    ax2.add_line(Line2D(line4_xs, line4_ys, linewidth=1, color='red',ls=':'))
    ax2.add_line(Line2D(line5_xs, line5_ys,linewidth=1,color='red',ls=':'))
    ax2.add_line(Line2D(line6_xs, line6_ys, linewidth=1, color='red',ls=':'))
    ax2.set_title('Color Warped Image', fontsize=20)
    # plt.show()

# HLS View
def visualize_HLS(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
    f.tight_layout()

#    ax1.imshow(img)
    ax1.set_title('Original', fontsize=20)

#    ax2.imshow(hls[:,:,0], cmap='gray')
    ax2.set_title('H', fontsize=20)

#    ax3.imshow(hls[:,:,1], cmap='gray')
    ax3.set_title('L', fontsize=20)

#    ax4.imshow(hls[:,:,2], cmap='gray')
    ax4.set_title('*S*', fontsize=20)
    # plt.show()
    return

# visualize_HLS(color_warped)

# HSV View
def visualize_HSV(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
    f.tight_layout()

#    ax1.imshow(img)
    ax1.set_title('Original', fontsize=20)

#    ax2.imshow(hsv[:,:,0], cmap='gray')
    ax2.set_title('H', fontsize=20)

#    ax3.imshow(hsv[:,:,1], cmap='gray')
    ax3.set_title('S', fontsize=20)

#    ax4.imshow(hsv[:,:,2], cmap='gray')
    ax4.set_title('*V*', fontsize=20)
    # plt.show()
    return

# visualize_HSV(color_warped)

# LUV View
def visualize_LUV(img):

    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
    f.tight_layout()

#    ax1.imshow(img)
    ax1.set_title('Original', fontsize=20)

#    ax2.imshow(luv[:,:,0], cmap='gray')
    ax2.set_title('*L*', fontsize=20)

#    ax3.imshow(luv[:,:,1], cmap='gray')
    ax3.set_title('U', fontsize=20)

#    ax4.imshow(luv[:,:,2], cmap='gray')
    ax4.set_title('V', fontsize=20)

    # plt.show()
    return

visualize_LUV(color_warped)

def visualize_binary_thresholded_image(img, combined_binary, b_binary, l_binary, show_title=False):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey='col', sharex='row', figsize=(15, 2.5))
    f.tight_layout(pad=1.0, w_pad=0.5, h_pad=2.0)

    if show_title: ax1.set_title('Warped Image', fontsize=16)
#    ax1.imshow(img)
    # plt.show()

    if show_title: ax2.set_title('B threshold', fontsize=16)
#    ax2.imshow(b_binary, cmap='gray')
    # plt.show()

    if show_title: ax3.set_title('L threshold', fontsize=16)
#    ax3.imshow(l_binary, cmap='gray')
    # plt.show()

    if show_title: ax4.set_title('Combined thresholds', fontsize=16)
#    ax4.imshow(combined_binary, cmap='gray')
    # plt.show()

    f.canvas.draw()
    data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(f.canvas.get_width_height()[::-1] + (3,))

    return data

for image_path in glob.glob(TEST_IMAGES_PATH):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undistorted_img = _camera_calibrator.undistort(image)
    color_warped = _perspective_transformer.warp_image(undistorted_img, M)
    binary_warped, binary_arr = ImageUtil.binary_thresholded_image(color_warped)
    # visualize_binary_thresholded_image(color_warped, binary_warped, binary_arr[0], binary_arr[1], True)

def visualize_peaks_in_binary_warped_image():
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.tight_layout(w_pad=4.0)
    ax1.set_title('Warped binary image', fontsize=16)
#    ax1.imshow(binary_warped, cmap='gray')
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    ax2.plot(histogram)
    ax2.set_title('Histogram of lower half image', fontsize=16)
    ax2.set_xlabel('Pixel positions')
    ax2.set_ylabel('Counts')
    # plt.show()

# visualize_peaks_in_binary_warped_image()

""" Visualize the polynormal fit of test images """
for image_path in glob.glob(TEST_IMAGES_PATH):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undistorted_img = _camera_calibrator.undistort(image)
    color_warped = _perspective_transformer.warp_image(undistorted_img, M)
    binary_warped, _ = ImageUtil.binary_thresholded_image(color_warped)
    _lane.polyfit_lines(binary_warped=binary_warped, visualize=False)#True)

left_fitx, right_fitx, ploty, curvrad, vpos =_lane.polyfit_lines(binary_warped, False)
text_curvrad = 'Radius of curvature: {}m'.format(curvrad)
vpos_direction = 'left' if vpos <binary_warped.shape[1]//2 else 'right'
text_vpos = 'Vehicle position:{:.2f}m {} of center'.format(vpos * 3.7/700,vpos_direction)
print(text_curvrad)
print(text_vpos)

def overlay_to_original_image(overlay_img, original_img):
    return cv2.addWeighted(original_img, 1, overlay_img, 0.3, 0)

left_fitx, right_fitx, ploty, radius_of_curvature, center = _lane.polyfit_lines(binary_warped, False)
overlay_img = _lane.overlay_image(
    binary_warped,
    original_img,
    Minv,
    left_fitx,
    right_fitx,
    ploty
)
result = overlay_to_original_image(overlay_img, original_img)
plt.figure()
##plt.imshow(result)
plt.title('Car lane area with original image')
# # plt.show()


# Pipeline
def init_process_video():
    """ Initialize global variables in preparation for processing image frames """
    global _lane
    _lane = CarLane()
    M, Minv = _perspective_transformer.transform_matrix(src, dst)


def process_frame(frame_img, debug=False):
    """ Main method for processing each image frame of a video file """
    text_output = []
    # Step1: Undistort image using the camera calibration
    undistorted_img = _camera_calibrator.undistort(frame_img)

    # Step2: Apply perspective transformation to create a warped image
    color_warped = _perspective_transformer.warp_image(undistorted_img, M)

    # Step3: Use color transforms, gradients, etc., to create a thresholded binary image
    binary_warped, binary_arr = ImageUtil.binary_thresholded_image(color_warped)
    image_shape = binary_warped.shape
    lane_detection_output = visualize_binary_thresholded_image(color_warped, binary_warped, binary_arr[0], binary_arr[1])

    # Step4: Get the sliding window polynomials for the left and right line
    leftx, lefty, rightx, righty = _lane.find_lane_pixels(binary_warped)

    ploty = np.linspace(0, image_shape[0] - 1, image_shape[0])


    left_fitx = _lane.left.polyfit_lines(leftx, lefty, image_shape)

    right_fitx = _lane.right.polyfit_lines(rightx, righty, image_shape)

    # Step5: Overlay the warped image to the original image
    overlay_img = _lane.overlay_image(binary_warped, frame_img, Minv, left_fitx, right_fitx, ploty)
    result = overlay_to_original_image(overlay_img, frame_img)

    # Step6: curvature_and_vehicle_position
    vehicle_position = _lane.get_vehicle_position(image_shape)
    radius_of_curvature = _lane.get_radius_of_curvature()

    text_output.append('Radius of curvature: {} m'.format(radius_of_curvature))
    text_output.append('Vehicle: {}'.format(vehicle_position))
    text_output.append('');
    text_output.append('Left line curve:')
    text_output.append('{:.2}, {:.2}, {:.2}'.format(_lane.left.line_fit0_queue[0], _lane.left.line_fit1_queue[0], _lane.left.line_fit2_queue[0]))
    text_output.append('');
    text_output.append('Right line curve:')
    text_output.append('{:.2}, {:.2}, {:.2}'.format(_lane.right.line_fit0_queue[0], _lane.right.line_fit1_queue[0], _lane.right.line_fit2_queue[0]))
    text_output.append('');
    if _lane.left.detected == False: text_output.append('!!!Left lane not detected!!!')
    if _lane.right.detected == False: text_output.append('!!!Right lane not detected!!!')

    console_img = ImageUtil.image_console(result, frame_img, undistorted_img, lane_detection_output, overlay_img,
                             text_output)
    return console_img

# Test
# Load test images
test_images = VideoUtil.images_from_video('project_video.mp4')

def test_image_frame():
    _, image = test_images[0]
    text_output = []
    # Step1: Undistort image using the camera calibration
    undistorted_img = _camera_calibrator.undistort(image)

    # Step2: Apply perspective transformation to create a warped image
    color_warped = _perspective_transformer.warp_image(undistorted_img, M)

    # Step3: Use color transforms, gradients, etc., to create a thresholded binary image
    binary_warped, binary_arr = ImageUtil.binary_thresholded_image(color_warped)
    image_shape = binary_warped.shape
    lane_detection_output = visualize_binary_thresholded_image(color_warped, binary_warped, binary_arr[0], binary_arr[1])

test_image_frame()

init_process_video()

def testVideoImages(images=fixed(test_images), i=1247):
    file, original_img = images[i]
    output = process_frame(original_img)
    plt.figure(figsize=(20,20))
#    #plt.imshow(output)

def processVideo(input_file_name, output_file_name):
    init_process_video()

    clip2 = VideoFileClip(input_file_name)
    #clip2.subclip(frame_start=5, frame_end=15).speedx(2) # works
    clip_handler = clip2.fl_image(process_frame)
    clip_handler.write_videofile(output_file_name, audio=False)

processVideo('project_video.mp4', 'project_video_result.mp4')
