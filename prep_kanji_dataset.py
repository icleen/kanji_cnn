import os
import sys
import cv2
import json
import h5py
import codecs
import numpy as np
from scipy import ndimage, misc
from PIL import Image, ImageDraw, ImageFont

def read_image(image_path):
    img = cv2.imread(image_path,1)
    return img
def grey(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
def resize_img(img,resize_dim):
    resized_img = cv2.resize(img, resize_dim, interpolation = cv2.INTER_LINEAR )
    return resized_img

def get_data(file_list, label_y_dict, size):
    x = np.zeros(shape=(len(file_list),size[0],size[1]), dtype=np.float32)
    y = []
    for i in range(len(file_list)):
        file = file_list[i]
        img = read_image(file)
        grey_image = grey(img)
        resized_image = resize_img(grey_image,size)
        x[i] = resized_image
        y_k = os.path.basename(file).split('_')[0]
        y.append(label_y_dict[y_k])
    # y = np.array(y,dtype='uint8').reshape((len(y),1))
    x = np.reshape(x, (len(file_list), size[0] * size[1]))
    x /= 255.0
    return x,y

def get_files_list(image_path):
    file_list = []
    folder_list = []
    for subdir, dirs, files in os.walk(image_path):
        folder_list.append(subdir)
        for file in files:
            file_path = os.path.join(subdir,file)
            file_list.append(file_path)
    return file_list, folder_list

def make_dictionary(folder_list):
    key = ''
    dictionary = {}
    for i, folder in enumerate(folder_list):
        key = os.path.basename(folder).split('/')[0]
        dictionary[key] = i

    return dictionary

def split_file(file_list):
    file_list =  np.array(file_list)
    np.random.shuffle(file_list)
    divide_indices = int(file_list.shape[0] * 0.85)
    train_file_list = file_list[0:divide_indices]
    test_file_list = file_list[divide_indices:]
    train_file_list = list(train_file_list)
    test_file_list = list(test_file_list)
    return train_file_list, test_file_list

def get_training_data():
    with h5py.File('training_data','r') as hf:
        training = np.array(hf.get('training'))
        t_labels = np.array(hf.get('t_labels'))
        validation = np.array(hf.get('validation'))
        v_labels = np.array(hf.get('v_labels'))
    return training, t_labels, validation, v_labels


def make_json():
    file_path = 'kanji_dataset/characters/'
    file_list = []
    folder_list = []
    for subdir, dirs, files in os.walk(image_path):
        folder_list.append(subdir)
        for file in files:
            file_path = os.path.join(subdir,file)
            file_list.append(file_path)

    dictionary = make_dictionary(folder_list)
    labels = []
    label = {}
    for file in file_list:
        gt = os.path.basename(file).split('_')[0]
        label = {"gt": dictionary[gt], "image_path": file}

if __name__ == '__main__':
    file_path = 'kanji_dataset/characters/'
    file_list, folder_list = get_files_list(file_path)
    classes = len(folder_list)
    size = (32, 32)
    dictionary = make_dictionary(folder_list)

    train_file_list, val_file_list = split_file(file_list)

    training, t_labels = get_data(train_file_list, dictionary, size)
    t_labels = np.asarray(t_labels, dtype=np.int32)
    print(training.shape, t_labels.shape)

    validation, v_labels = get_data(val_file_list, dictionary, size)
    v_labels =  np.asarray(v_labels, dtype=np.int32)
    print(validation.shape, v_labels.shape)

    with h5py.File('training_data', 'w') as hf:
         hf.create_dataset('training', data = training)
         hf.create_dataset('t_labels', data = t_labels)
         hf.create_dataset('validation', data = validation)
         hf.create_dataset("v_labels", data = v_labels)
