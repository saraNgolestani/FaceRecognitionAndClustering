import cv2
import os
import shutil


class SizeUtils:
    '''
    The ResizeUtils provides resizing function to keep the aspect ratio intact
    Credits: AndyP at StackOverflow'''

    # Given a target height, adjust the image
    # by calculating the width and resize
    def rescale_by_height(self, image, target_height,
                          method=cv2.INTER_LANCZOS4):
        # Rescale `image` to `target_height`
        # (preserving aspect ratio)
        w = int(round(target_height * image.shape[1] / image.shape[0]))
        return (cv2.resize(image, (w, target_height),
                           interpolation=method))

    # Given a target width, adjust the image
    # by calculating the height and resize
    def rescale_by_width(self, image, target_width,
                         method=cv2.INTER_LANCZOS4):
        # Rescale `image` to `target_width`
        # (preserving aspect ratio)
        h = int(round(target_width * image.shape[0] / image.shape[1]))
        return (cv2.resize(image, (target_width, h),
                           interpolation=method))


class FramesGenerator:
    def __init__(self, VideoFootageSource):
        self.VideoFootageSource = VideoFootageSource

    # Resize the given input to fit in a specified
    # size for face embeddings extraction
    def AutoResize(self, frame):
        resizeUtils = SizeUtils()

        height, width, _ = frame.shape

        if height > 500:
            frame = resizeUtils.rescale_by_height(frame, 500)
            self.AutoResize(frame)

        if width > 700:
            frame = resizeUtils.rescale_by_width(frame, 700)
            self.AutoResize(frame)

        return frame

