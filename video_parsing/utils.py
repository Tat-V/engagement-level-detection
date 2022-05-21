import os
import shutil

import cv2
import numpy as np

# from video_parsing.facial_analysis import FacialImageProcessing
from tqdm import tqdm


def images_from_video(video_path, img_height, img_width, img_dir):
    try:
        shutil.rmtree(img_dir, ignore_errors=True)
        os.mkdir(img_dir)
    except:
        True

    img_list = []

    cap = cv2.VideoCapture(video_path)

    i = 0
    while True:
        success, image = cap.read()
        if success:
            image = cv2.resize(image, (img_height, img_width))
            img_list.append(image)
            cv2.imwrite(os.path.join(img_dir, f"img_{i}.jpg"), image)
            # print(os.path.join(img_dir, f'img_{i}.png'), image)
            i += 1
        else:
            print("Defected capture")
            return img_list
