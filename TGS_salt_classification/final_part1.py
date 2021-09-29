#Import necessary packages
import time
t_start = time.time()

import os
import sys
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style('white')

from sklearn.model_selection import train_test_split

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model, save_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from keras.layers import Input, Dropout, BatchNormalization, Activation, Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

from keras import backend as K
from keras import optimizers

import tensorflow as tf

from tqdm import tqdm

#Custom imports
from yputils.nn.unet_resnet import Unet_Resnet
from yputils.metrics.iou_metric import my_iou_metric
from yputils.callbacks.sgdScheduler import SGDRScheduler
from yputils.plots.plot_history import plot_history1
from yputils.predict.predict_results import predict_aug_result


#Save some file names and variable names
version = 2
basic_name = f'Unet_resnet_v{version}_BCE'
save_model_name = basic_name + '.model'
submission_file = basic_name + '.csv'

#Load train and test ids
train_df = pd.read_csv("./data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("./data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

#Add the image and mask data to the corresponding ids in df
#*****************************check the location before upload
train_df["images"] = [np.array(load_img("./data/train/images/{}.png".format(idx),\
                grayscale=True)) / 255 for idx in tqdm(train_df.index)]
train_df["masks"] = [np.array(load_img("./data/train/masks/{}.png".format(idx),\
                grayscale=True)) / 255 for idx in tqdm(train_df.index)]

#Extract the X_train and y_train data
x_train = np.array(train_df.images.tolist()).reshape(-1, 101, 101, 1)
y_train = np.array(train_df.masks.tolist()).reshape(-1, 101, 101, 1)

#Data augmentation
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)


#Build the model
input_layer = Input((101, 101, 1))
output_layer = Unet_Resnet.build(input_layer, 32,0.5)

model1 = Model(input_layer, output_layer)

#Compile the model with binary_crossentropy
c = optimizers.adam(lr = 0.005)
model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])

#Construct the callbacks
early_stopping = EarlyStopping(monitor='my_iou_metric', mode = 'max',patience=15, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name, monitor='my_iou_metric', mode='max',
                                   save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode='max', factor=0.5, patience=5,
                              min_lr=0.0001, verbose=1)

epoch_size = len(x_train)
epochs = 110
batch_size = 2

schedule = SGDRScheduler(min_lr=1e-5,
                        max_lr=1e-3,
                        steps_per_epoch=np.ceil(epoch_size/batch_size),
                        lr_decay=0.9,
                        cycle_length=5,
                        mult_factor=1.5)

#Train the model
t_model1_start = time.time()
history = model1.fit(x_train, y_train,
                     epochs = epochs,
                     batch_size = batch_size,
                     callbacks = [ schedule,model_checkpoint, reduce_lr],
                     verbose = 1)
t_model1_end = time.time()
print(f"Run time = {(t_model1_end-t_model1_start)/3600} hours")

#plot the convergence
plot_history1(history, "loss", "my_iou_metric")

#Load the trained model and predict the images
'''
The code till above is only required because we will use the
above saved model as initialization for lovsaz optimization
'''
model = load_model(save_model_name,\
                custom_objects={'my_iou_metric':my_iou_metric})

preds_valid = predict_aug_result(model,x_train,img_size_target=101)


#
