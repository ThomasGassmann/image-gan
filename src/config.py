import json
import glob
import os
import ntpath
import cv2
import sys
from PIL import Image

class Configuration:
    def __init__(self, config = 'config.json'):
        file_data = open(config).read()
        self.data = json.loads(file_data)

    def get_random_dim(self):
        return int(self.data['random_dim'])

    def load_data(self):
        image_dir = self.data['image_directory']
        resized_image_dir = self.data['resized_image_directory']
        height = self.data['resolution']['height']
        width = self.data['resolution']['width']
        rgba2rgb = bool(self.data['rgba2rgb'])
        if not os.path.exists(image_dir):
            raise Exception('Source path does not exist')

        files = glob.glob(image_dir + '/**/*.jpg', recursive=True)

        if not os.path.exists(resized_image_dir):
            os.makedirs(resized_image_dir)
        
        for file in files:
            file_name = ntpath.basename(file)
            destination_path = os.path.join(resized_image_dir, file_name)
            if not os.path.isfile(destination_path):
                img = cv2.imread(file)
                img = cv2.resize(img, (width, height))
                cv2.imwrite(destination_path, img)
                if rgba2rgb:
                    self.__rgb2rgba(destination_path)

        return glob.glob(resized_image_dir + '/**/*.jpg', recursive=False)

    def __rgb2rgba(self, img_path):
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img.load()
            image_background = Image.new('RGB', jpg.size, (0, 0, 0))
            image_background.paste(img, mask=img.split()[3])
            img = image_background
        else:
            img.convert('RGB')
            
        img.save(img_path, 'JPEG')
