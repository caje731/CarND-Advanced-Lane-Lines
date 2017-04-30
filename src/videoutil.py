#!/usr/bin/python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

class VideoUtil:

    @staticmethod
    def images_from_video(file_name):
        """ Convert a video file to image frames and return the image frames """
        count = 0

        def convert_frame(img):
            nonlocal count
            f = "%s - %d" % (file_name, count)
            count += 1
            return f, img

        clip = VideoFileClip(file_name)
        return [convert_frame(frame) for frame in clip.iter_frames(progress_bar=True)]

    @staticmethod
    def save_images(file_name, save_path):
        """ Convert a video file to image frames and return the image frames """
        count = 0

        def save_frame(img):
            nonlocal count
            f = "%s/%s_%d.png" % (save_path, file_name, count)
            count += 1
            plt.imsave(f, img)

        clip = VideoFileClip(file_name)
        for frame in clip.iter_frames(progress_bar=True):
            save_frame(frame)