import json
import glob
import os
import ntpath
import cv2
import sys
from PIL import Image
import numpy as np

def resize(directory):
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        image = cv2.imread(full_path)
        print('Resizing: ' + full_path)
        resized_image = cv2.resize(image, (256, 256))
        cv2.imwrite(full_path, resized_image)


def toRGB(directory):
    files = os.listdir(directory)
    for file in files:
        full_path = os.path.join(directory, file)
        img = Image.open(full_path)
        if img.mode == 'RGBA':
            img.load()
            image_background = Image.new('RGB', jpg.size, (0, 0, 0))
            image_background.paste(img, mask=img.split()[3])
            img = image_background
        else:
            img.convert('RGB')        
        img.save(full_path, 'JPEG')


def load_data(directory):
    return_values = []
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        image = cv2.imread(full_path)
        array = np.asarray(image)
        combined_rgb = np.average(array, axis=2)
        rounded = np.rint(combined_rgb)
        return_values.append(rounded)
    return np.array(return_values)
