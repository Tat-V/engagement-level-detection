import glob
import pathlib

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn import metrics, preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from tensorflow.keras.applications import (
    densenet,
    inception_resnet_v2,
    inception_v3,
    mobilenet,
    mobilenet_v2,
    resnet,
    resnet_v2,
    vgg16,
)
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Activation,
    BatchNormalization,
    Conv2D,
    Conv3D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPool2D,
    GlobalMaxPool3D,
    Input,
    Masking,
    MaxPool2D,
    MaxPooling2D,
    MaxPooling3D,
    Reshape,
    TimeDistributed,
)
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.models import Model, Sequential, load_model, model_from_json
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2 as L2_reg
from tensorflow.keras.utils import to_categorical

# from tensorflow.compat.v1.keras.backend import set_session


print(tf.__version__)

import json
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
from skimage import transform as trans
from tqdm import tqdm

from prediction.utils import *
from video_parsing import FacialImageProcessing

emotion_to_index = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Surprise": 6,
}
idx_to_class = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Neutral",
    5: "Sadness",
    6: "Surprise",
}

INPUT_SIZE = (224, 224)
BATCH_SIZE = 40  # 512 #64 #32 #64

MODEL = load_model("../models/affectnet_emotions_mobilenet_7.h5")
IMGS_PATH = "../resources/images_from_video/"


def predict_emotion(imgs_path, model=MODEL):
    frame_bgr = cv2.imread(imgs_path)
    # plt.figure(figsize=(5, 5))
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # plt.imshow(frame)
    img_processing = FacialImageProcessing(False)
    bounding_boxes, points = img_processing.detect_faces(frame)
    points = points.T
    scores = []
    if not (bounding_boxes == [] or points == []):
        for bbox, p in zip(bounding_boxes, points):
            box = bbox.astype(np.int)
            x1, y1, x2, y2 = box[0:4]
            face_img = frame[y1:y2, x1:x2, :]

            face_img = cv2.resize(face_img, INPUT_SIZE)
            inp = face_img.astype(np.float32)
            inp[..., 0] -= 103.939
            inp[..., 1] -= 116.779
            inp[..., 2] -= 123.68
            inp = np.expand_dims(inp, axis=0)
            scores = model.predict(inp)[0]
            # plt.figure(figsize=(3, 3))
            # plt.imshow(face_img)
            # plt.title(idx_to_class[np.argmax(scores)])
            # print(idx_to_class[np.argmax(scores)])
    return scores


def make_emotions_dataset(imgs_path=IMGS_PATH):
    preds_file = "../resources/prediction_answer.csv"
    for i in glob.glob(imgs_path + "*.jpg"):
        scores = predict_emotion(i)
        if len(scores) != 0:
            with open(preds_file, "a+") as f:
                f.write(",".join(map(str, scores)) + "\n")
