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
from yputils.metrics.iou_metric import my_iou_metric,my_iou_metric_2
from yputils.callbacks.sgdScheduler import SGDRScheduler
from yputils.plots.plot_history import plot_history1
from yputils.predict.predict_results import predict_aug_result
from yputils.loss_fn.lovasz_loss import lovasz_loss
from yputils.metrics.iou_threshold import iou_threshold_batch
from yputils.scores.rle_encoding import rle_encode


#Save some file names and variable names
version = 2
basic_name = f'Unet_resnet_v{version}_lovasz'
pretrained_model = f'Unet_resnet_v{version}_BCE.model'
save_model_name = basic_name + '.model'
 = basic_name + '.csv'

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

# remove activation layer to use lovasz loss
input_x = model1.layers[0].input
output_layer = model1.layers[-1].input
model2 = Model(input_x, output_layer)

#Compile the model with Lovasz loss
c = optimizers.adam(lr = 0.001)
model2.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

#Load the weights from BCE model- model1
#*****************************check the file paths
model2.load_weights(pretrained_model, by_name=True)

#Construct the callbacks
early_stopping = EarlyStopping(monitor='my_iou_metric_2', mode = 'max',patience=30, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='my_iou_metric_2',
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric_2', mode = 'max',factor=0.5, patience=5,
                              min_lr=0.00005, verbose=1)

epochs = 110
batch_size = 2

schedule = SGDRScheduler(min_lr=1e-5,
                        max_lr=1e-3,
                        steps_per_epoch=np.ceil(epoch_size/batch_size),
                        lr_decay=0.9,
                        cycle_length=5,
                        mult_factor=1.5)

#Train the model
t_model2_start = time.time()
history = model2.fit(x_train, y_train,
                    #validation_data=[x_valid, y_valid],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping],
                    verbose=2)
t_model2_end = time.time()
print(f"Run time = {(t_model2_end-t_model2_start)/3600} hours")

#plot the convergence
plot_history1(history, "loss", "my_iou_metric_2")

#Load the trained model and predict the images
model = load_model(save_model_name,\
                custom_objects={'my_iou_metric_2':my_iou_metric_2,\
                                'lovasz_loss': lovasz_loss})

preds_valid = predict_aug_result(model,x_train,img_size_target=101)

#Score the model and do a threshold optimization by the best IoU

## Scoring for last model, choose threshold by validation data
thresholds_ori = np.linspace(0.3, 0.7, 31)
# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori))

ious = np.array([iou_threshold_batch(y_train, preds_valid > threshold) \
                for threshold in tqdm(thresholds)])
print("The list of ious for given thresholds, \n", ious)

#Get the index of best iou value
threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

#Plot the iou vs threshold
plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, 'xr', label='Best threshold')
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()

#Load the test Data
x_test = np.array([(np.array(load_img("./data/test/images/{}.png".format(idx), grayscale = True))) / 255\
        for idx in tqdm(test_df.index)]).reshape(-1, 101, 101, 1)

#Predict the results on the test Data
preds_test = predict_result(model,x_test,101)

#rle rle_encoding
pred_dict = {idx: rle_encode(np.round(preds_test[i] > threshold_best))\
            for i, idx in enumerate(tqdm(test_df.index.values))}

#Save the submission file
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(submission_file)

#Run time
t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")











#
