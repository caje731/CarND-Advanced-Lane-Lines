import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageUtil():

    @staticmethod
    def image_console(mainScreen, sc1, sc2, sc3, sc4, text_arr):

        font = cv2.FONT_HERSHEY_COMPLEX
        textpanel = np.zeros((240, 640, 3), dtype=np.uint8)
        text_pos_y = 30
        for text in text_arr:
            cv2.putText(textpanel, text, (10, text_pos_y), font, 1, (255, 255, 255), 2)
            text_pos_y += 30

        # assemble the screen example
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        if mainScreen is not None: canvas[0:720, 0:1280] = mainScreen

        if sc2 is not None: canvas[0:240, 1280:1600] = cv2.resize(sc2, (320, 240), interpolation=cv2.INTER_AREA)
        if sc4 is not None: canvas[0:240, 1600:1920] = cv2.resize(sc4, (320, 240), interpolation=cv2.INTER_AREA)

        canvas[240:480, 1280:1920] = textpanel
        if sc3 is not None: canvas[720:1080, 0:1280] = cv2.resize(sc3, (1280, 360), interpolation=cv2.INTER_AREA) * 4

        return canvas


    @staticmethod
    def binary_thresholded_image(img, b_thresh_min=145, b_thresh_max=200, \
                                 l_thresh_min=215, l_thresh_max=255):
        """ Return the binary Thresholded image by combining multiple binary thresholds  """
        l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]  # Detect white lines
        b_channel = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]  # Detect yellow lines

        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

        combined_binary = np.zeros_like(b_binary)
        combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

        return combined_binary, (l_binary, b_binary)