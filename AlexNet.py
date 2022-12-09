import tflearn
from tflearn import local_response_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import Data_Preparation

X_train, Y_train, X_test, Y_test = Data_Preparation.get_Dataset(image_size=227, isRGB=1)

conv_input = input_data(shape=[None, 227, 227, 3], name='input')

conv1 = conv_2d(conv_input, 96, 11, strides=4, activation='relu')
pool1 = max_pool_2d(conv1, 3, strides=2)
pool1 = local_response_normalization(pool1)
conv2 = conv_2d(pool1, 256, 5, strides=1, activation='relu')
pool2 = max_pool_2d(conv2, 3, strides=2)
pool2 = local_response_normalization(pool2)
conv3 = conv_2d(pool2, 384, 3, strides=1, activation='relu')
conv4 = conv_2d(conv3, 384, 3, strides=1, activation='relu')
conv5 = conv_2d(conv4, 256, 3, strides=1, activation='relu')
pool3 = max_pool_2d(conv5, 3, strides=2)
pool3 = local_response_normalization(pool3)
fully_layer = fully_connected(pool3, 4096, activation='tanh')
dropout1 = dropout(fully_layer, 0.5)
fully_layer2 = fully_connected(dropout1, 4096, activation='tanh')
dropout2 = dropout(fully_layer2, 0.5)
output_layer = fully_connected(dropout2, 6, activation='softmax')

alexNet = regression(output_layer, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                     name='targets')

model = tflearn.DNN(alexNet, tensorboard_dir='log', tensorboard_verbose=2)

model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=1000,
          validation_set=({'input': X_test}, {'targets': Y_test}),
          snapshot_step=200, batch_size=64, show_metric=True, run_id='alexNet-Sports')

model.save('alexNet.tfl')
