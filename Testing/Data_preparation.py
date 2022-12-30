import os
from random import shuffle
import cv2
import numpy as np
import Data_Preprocessing

data_path = '../data/'


def get_Dataset(image_size, isRGB):
    test_data = []
    for type_folder in os.listdir(data_path):
        if type_folder == 'Exam':
            test_data = read_images(data_path + type_folder, isRGB, image_size)

    X_test, Y_test = reformat_dataset(test_data, image_size, isRGB)

    test_images_names = get_images_name(test_data)
    return X_test, test_images_names


def read_images(images_paths, isRGB, image_size):
    images = []
    for i in os.listdir(images_paths):
        image = cv2.imread(os.path.join(images_paths, i), isRGB)
        image = resize_image(image, image_size)
        image_label = create_label(i)
        images.append([np.array(image), image_label, i])
    return images


def resize_image(image, image_size):
    return cv2.resize(image, (image_size, image_size))


def create_label(image_path):
    image_label = image_path.split('_')[0]
    image_classes = ['Basketball', 'Football', 'Rowing', 'Swimming', 'Tennis', 'Yoga']
    label_encoded = np.zeros((6, 1))
    for i in range(len(image_classes)):
        if image_label == image_classes[i]:
            label_encoded[i] = 1
    return label_encoded


def get_images_name(data):
    names = []
    for i in data:
        names.append(i[2])
    return names


def reformat_dataset(data, image_size, isRGB):
    if isRGB == 0:
        X = np.array([i[0] for i in data], dtype=object).reshape(-1, image_size, image_size, 1)
    else:
        X = np.array([i[0] for i in data], dtype=object).reshape(-1, image_size, image_size, 3)

    Y = np.array([i[1] for i in data])
    Y = Y.reshape(len(Y), 6)
    return X, Y
