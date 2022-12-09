import os

import tflearn
from tflearn import input_data, conv_2d, max_pool_2d, fully_connected, dropout, regression

import testingScript
import Data_Preparation

X_train, Y_train, X_test, test_images_names = Data_Preparation.get_Dataset(image_size=50, isRGB=0)

conv_input = input_data(shape=[None, 50, 50, 1], name='input')

conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 5, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)

fully_layer = fully_connected(pool5, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                        name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('TrainedModels/labArchitectureModel/labModel.tfl.meta'):
    model.load('./TrainedModels/labArchitectureModel/labModel.tfl')
else:
    model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=100,
              snapshot_step=500, show_metric=True, run_id='sports')
    model.save('labModel.tfl')

testingScript.generateTestingCSV(model, X_test, test_images_names)
