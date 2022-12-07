import os
import cv2
import numpy as np

data_path = 'data/'


# Function to get Data Set
def get_Dataset():
    train_data = []
    test_data = []
    for type_folder in os.listdir(data_path):
        if type_folder == 'Train':
            train_data = read_images(data_path + type_folder)
        else:
            test_data = read_images(data_path + type_folder)
    X_train, Y_train = reformat_dataset(train_data)
    X_test, Y_test = reformat_dataset(test_data)
    return X_train, Y_train, X_test, Y_test


# Function to read images with their path
def read_images(images_paths):
    images = []
    for i in os.listdir(images_paths):
        image = cv2.imread(os.path.join(images_paths, i), 0)
        image = resize_image(image, 227)
        image_label = create_label(i)
        images.append([np.array(image), image_label])
    return images


# Function to resize the Images
def resize_image(image, image_size):
    return cv2.resize(image, (image_size, image_size))


# Function to Labelled Images
def create_label(image_path):
    image_label = image_path.split('_')[0]
    image_classes = ['Basketball', 'Football', 'Rowing', 'Swimming', 'Tennis', 'Yoga']
    label_encoded = np.zeros((6, 1))
    for i in range(len(image_classes)):
        if image_label == image_classes[i]:
            label_encoded[i] = 1
    return label_encoded


# Function to reformat the Data Set
def reformat_dataset(data):
    X = np.array([i[0] for i in data], dtype=object)
    Y = np.array([i[1] for i in data])
    return X, Y


