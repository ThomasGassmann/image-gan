import json
import glob
import os
import ntpath
import cv2
import sys
from PIL import Image
import numpy as np


def load_data(directory):
    return_value = []
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        image = cv2.imread(full_path)
        resized_image = cv2.resize(image, (28, 28))
        img = __rgb_transcode(resized_image)
        return_value.append(img)
    return np.array(return_value)

  
def __rgb_transcode(image):
    array = []
    for x in image:
        x_array = []
        for y in x:
            avg = int(sum(y) / len(y))
            x_array.append(avg)
        array.append(x_array)
    return array