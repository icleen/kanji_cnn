import os
import sys
import cv2
import json
import h5py
import codecs
import numpy as np
from scipy import ndimage, misc
from PIL import Image, ImageDraw, ImageFont

def get_training_data():
    with h5py.File('training_data','r') as hf:
        training = np.array(hf.get('training'))
        t_labels = np.array(hf.get('t_labels'))
        validation = np.array(hf.get('validation'))
        v_labels = np.array(hf.get('v_labels'))
    return training, t_labels, validation, v_labels

if __name__ == '__main__':
    data = get_training_data()
    training = data[0]
    print(training.shape)
    print(training[0].shape)
    print(training[0])
