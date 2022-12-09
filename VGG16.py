import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import Data_Preparation

X_train, Y_train, X_test, Y_test = Data_Preparation.get_Dataset(image_size=227, isRGB=1)

conv_input = input_data(shape=[None, 224, 224, 3], name='input')

conv1 = conv_2d(conv_input, 64, 3, strides=1, activation='relu')
conv2 = conv_2d(conv1, 64, 3, strides=1, activation='relu')

pool1 = max_pool_2d(conv2, kernel_size=2, strides=2)

conv3 = conv_2d(pool1, 128, 3, padding="same", activation="relu")
conv4 = conv_2d(conv3, 128, 3, padding="same", activation="relu")

pool2 = max_pool_2d(conv4, 2, strides=2)

conv5 = conv_2d(pool2, 256, 3, padding="same", activation="relu")
conv6 = conv_2d(conv5, 256, 3, padding="same", activation="relu")
conv7 = conv_2d(conv6, 256, 3, padding="same", activation="relu")

pool3 = max_pool_2d(conv7, 2, strides=2)

conv8 = conv_2d(pool3, 512, 3, padding="same", activation="relu")
conv9 = conv_2d(conv8, 512, 3, padding="same", activation="relu")
conv10 = conv_2d(conv9, 512, 3, padding="same", activation="relu")

pool4 = max_pool_2d(conv10, 2, strides=2)

conv11 = conv_2d(pool4, 512, 3, padding="same", activation="relu")
conv12 = conv_2d(conv11, 512, 3, padding="same", activation="relu")
conv13 = conv_2d(conv12, 512, 3, padding="same", activation="relu")

pool5 = max_pool_2d(conv13, 2, strides=2, name='vgg16')

fully_layer1 = fully_connected(pool5, 256, activation='relu')
fully_layer2 = fully_connected(fully_layer1, 128, activation='relu', name='fc2')
dropout2 = dropout(fully_layer2, 0.5)
output_layer = fully_connected(dropout2, 6, activation='softmax')

VGG16 = regression(output_layer, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                   name='targets')

model = tflearn.DNN(VGG16, tensorboard_dir='log', tensorboard_verbose=2)

model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=1000,
          validation_set=({'input': X_test}, {'targets': Y_test}),
          snapshot_step=200, batch_size=64, show_metric=True, run_id='alexNet-Sports')

model.save('VGG16.tfl')
