import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = '../data/stage1_train/'
TEST_PATH = '../data/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Build model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

d1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
d1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (d1)
p1 = MaxPooling2D((2, 2)) (d1)

d2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
d2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (d2)
p2 = MaxPooling2D((2, 2)) (d2)

d3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
d3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (d3)
p3 = MaxPooling2D((2, 2)) (d3)

d4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
d4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (d4)
p4 = MaxPooling2D((2, 2)) (d4)

d5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
d5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (d5)
u4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (d5)

u4 = concatenate([u4, d4])
u4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u4)
u4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u4)
u3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (u4)

u3 = concatenate([u3, d3])
u3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u3)
u3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u3)
u2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (u3)

u2 = concatenate([u2, d2])
u2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u2)
u2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u2)
u1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (u2)

u1 = concatenate([u1, d1])
u1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u1)
u1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u1)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (u1)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

checkpointer = ModelCheckpoint('checkpoint', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=16, epochs=100, callbacks=[checkpointer])
