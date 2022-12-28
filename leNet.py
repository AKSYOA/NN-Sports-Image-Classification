import os

import tflearn
from tflearn import input_data, conv_2d, max_pool_2d, fully_connected, dropout, regression, avg_pool_2d

import CSV_utilities
import Data_Preparation

X_train, Y_train, X_test, test_images_names = Data_Preparation.get_Dataset(image_size=32, isRGB=0)

conv_input = input_data(shape=[None, 32, 32, 1], name='input')
conv1 = conv_2d(conv_input, 6, 5, strides=1, activation='relu', padding='same')
pool1 = avg_pool_2d(conv1, 2, strides=2)
conv2 = conv_2d(pool1, 16, 5, strides=1, activation='relu')
pool2 = avg_pool_2d(conv2, 2, strides=2)
conv3 = conv_2d(pool2, 120, 5, strides=1, padding='valid', activation='relu')
fully_connected2 = fully_connected(conv3, 84, activation='relu')
leNet = fully_connected(fully_connected2, 6, activation='softmax')

leNet = regression(leNet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                   name='targets')
model = tflearn.DNN(leNet, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('TrainedModels/leNet/leNet.tfl.meta'):
    model.load('./TrainedModels/leNet/leNet.tfl')
else:
    model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=26,
              snapshot_step=500, show_metric=True, run_id='sports')
    model.save('leNet.tfl')

CSV_utilities.generateTestingCSV(model, X_test, test_images_names)
