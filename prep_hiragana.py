import os
import numpy as np
import json
import codecs
from scipy import ndimage, misc
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

class prepper:

    def __init__(self, imageFolder, dictionary):
        self.image_root = imageFolder
        with open("training.json", 'r') as f:
            self.training = json.load(f)
        with open("validation.json", 'r') as f:
            self.validation = json.load(f)
        with codecs.open(dictionary, encoding="utf8") as f:
            text_lines = f.readlines()

        text_lines = [t.strip() for t in text_lines]
        text_lines = [t for t in text_lines if len(t) > 0]
        self.dictionary = {}
        for i, t in enumerate(text_lines):
            self.dictionary[t] = str(i)

    def train_images(self):
        train = np.ndarray(shape=(len(self.training), 32, 32), dtype=np.float32)
        img = []
        total = 0
        for k, obj in enumerate(self.training):
            img_path = self.image_root + obj['image_path']
            img = ndimage.imread(img_path)
            for i, row in enumerate(img):
                for j, pixels in enumerate(row):
                    total = float(int(pixels[0]) + int(pixels[1]) + int(pixels[2]))/3.0
                    train[k][i][j] = float(total/255.0)

        train = np.reshape(train, (len(self.training), 1024))
        return train

    def train_labels(self):
        labels = []
        for obj in self.training:
            gt = obj['gt']
            labels.append(self.dictionary[gt])

        return labels

    def validate_images(self):
        train = np.ndarray(shape=(len(self.validation), 32, 32), dtype=np.float32)
        img = []
        total = 0
        for k, obj in enumerate(self.validation):
            img_path = self.image_root + obj['image_path']
            img = ndimage.imread(img_path)
            for i, row in enumerate(img):
                for j, pixels in enumerate(row):
                    total = float(int(pixels[0]) + int(pixels[1]) + int(pixels[2]))/3.0
                    train[k][i][j] = float(total/255.0)

        train = np.reshape(train, (len(self.validation), 1024))
        return train

    def validate_labels(self):
        labels = []
        for obj in self.validation:
            gt = obj['gt']
            labels.append(self.dictionary[gt])

        return labels


if __name__ == "__main__":
    prepper = prepper('hiragana', 'hiragana.txt')
    train_data = prepper.train_images()
    print(train_data[0])
