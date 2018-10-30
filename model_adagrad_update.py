# -*- coding: utf-8 -*-
from __future__ import print_function
import random

import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,Adam, Adagrad, rmsprop
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras.layers.normalization import BatchNormalization

from keras import optimizers

from sklearn.preprocessing import MinMaxScaler



#수업 모듈
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt



from boss_input import extract_data, resize_with_pad, IMAGE_SIZE


class Dataset(object):

    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None

    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        images, labels = extract_data('../DATA/')   #     labels : boss 이면 0 , 아니면 1
        # print(images.shape)          #(467,64,64,3)
        # print(labels.shape)          #(467,)
        labels = np.reshape(labels, [-1])  # 굳이 할 필요없을 듯
        # print(labels.shape)          #(467,)

        # numpy.reshape
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=random.randint(0, 100))  # random.randint(0, 100) 0과 100사이의 랜덤한 정수를 생성
        X_valid, X_test, y_valid, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=random.randint(0, 100))

        # print('#####',X_train)
        # print('#####',X_test)
        # print('#####', X_valid)

        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
            X_valid = X_valid.reshape(X_valid.shape[0], 3, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
            X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 3)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

        # the data, shuffled and split between train and test sets
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)   #원핫인코딩
        Y_valid = np_utils.to_categorical(y_valid, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        print(X_train.dtype)
        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')



        # X_train /= 255  #데이터 정규화를 위해 ,   255:  0부터 255사이의 이미지 픽셀 값
        # X_valid /= 255
        # X_test /= 255

        # print('###',X_train)  #정규화 된 data 확인

        self.X_train = X_train  # 정규화 된 data
        self.X_valid = X_valid
        self.X_test = X_test
        self.Y_train = Y_train  # 원핫 인코딩된 label
        self.Y_valid = Y_valid
        self.Y_test = Y_test



class Model(object):

    FILE_PATH = '../store/model.h5'

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=2):


        #모델의 기본 오브젝트 생성  --> add함수를 이용하여 모델에 추가하는 것이 가능하다.
        self.model = Sequential()

        #합성곱층의 추가
        # border_mode='same' : 제로패딩을 지시하는 인수 . 스트라이드가 1이면 입력데이터와 특성지도의 크기가 같아지도록 조정
        # input_shape=  : (채널,세로,가로)

        # 1층
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=dataset.X_train.shape[1:]))
        #self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=dataset.X_train.shape[1:], activation='relu'))
        self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        self.model.add(Dropout(0.25))


        #2층
        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
        self.model.add(BatchNormalization(mode=0,axis=1))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))


        # 1차원화층을 모델에 추가
        self.model.add(Flatten())
        #전결합층을 모델에 추가 , 입력512(이전 레이어 입력)  출력 512
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(self, dataset, batch_size=32, epochs =20, data_augmentation=True):
        # let's train the model using SGD + momentum (how original).
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



        self.model.compile(loss='categorical_crossentropy',
                           optimizer= Adagrad(lr=0.001, epsilon=1e-6),
                           metrics=['accuracy'])
        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(dataset.X_train, dataset.Y_train,
                           batch_size=batch_size,
                           epochs=epochs ,
                           validation_data=(dataset.X_valid, dataset.Y_valid),
                           shuffle=True)
        else:
            print('Using real-time data augmentation.')

            # this will do preprocessing and realtime data augmentation
            # 데이터 증가를 통한 성능향상

            datagen = ImageDataGenerator(
                featurewise_center=False,             # set input mean to 0 over the dataset
                samplewise_center=False,              # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,   # divide each input by its std
                zca_whitening=False,                  # apply ZCA whitening
                rotation_range=20,                     # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.2,                # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,               # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,                 # randomly flip images
                vertical_flip=False,                # randomly flip images
                zoom_range=True,                    # 랜덤하게 이미지를 확대/축소
                fill_mode='nearest')               # 회전이나 시프트로 인해 새로 생긴 픽셀을 채우는 데 사용할 방법 정의.


            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)

            from keras.callbacks import EarlyStopping
            early_stopping = EarlyStopping()

            datagen.fit(dataset.X_train)

            # fit the model on the batches generated by datagen.flow()
            self.model.fit_generator(datagen.flow(dataset.X_train, dataset.Y_train,
                                                  batch_size=batch_size),
                                     steps_per_epoch=dataset.X_train.shape[0],
                                     epochs=epochs ,
                                     validation_data=(dataset.X_valid, dataset.Y_valid),
                                     callbacks=[early_stopping])

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    # def predict(self, image):
    #     if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
    #         image = resize_with_pad(image)
    #         image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    #     elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
    #         image = resize_with_pad(image)
    #         image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    #     image = image.astype('float32')
    #     image /= 255
    #     result = self.model.predict_proba(image)
    #     # print('!!!',result)
    #     result = self.model.predict_classes(image)
    #     # print('!!!',result)
    #
    #     return result[0]

    def evaluate(self, dataset,batch_size=10):

        score = self.model.evaluate(dataset.X_test, dataset.Y_test, batch_size=batch_size,verbose=2)
        # print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        print('Test Loss and Accuracy -> {:.2f},{:.2f}'.format(*score))

if __name__ == '__main__':
    dataset = Dataset()
    dataset.read()

    model = Model()
    print('@@@')
    model.build_model(dataset)
    print('@@@model train')
    model.train(dataset, epochs =1)
    model.save()

    model = Model()
    model.load()
    model.evaluate(dataset,batch_size=100)

    # print(dataset.X_test)
    # model.predict(dataset.X_test)  오류 , boss_input 모듈/resize_with_pad함수/get_padding_size함수/   h, w, _ = image.shape

##########################################
