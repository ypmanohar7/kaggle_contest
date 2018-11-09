#===============================================================================
# 				USAGE
#
# python lenet_mnist.py -s 1 -w lenet_weights.hdf5
# python lenet_mnist.py -l 1 -w lenet_weights.hdf5
#===============================================================================

#===============================================================================
# 			import the necessary packages
#===============================================================================
from yputils.cnn.networks.lenet import LeNet
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#===============================================================================
# 		construct the argument parse and parse the arguments
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

#===============================================================================
# 		Segregate the train and test data
#===============================================================================
train_df = pd.read_csv("./data/train.csv")
test_final = pd.read_csv("./data/test.csv")

Y_train = train_df['label']
X_train = train_df.drop(labels=['label'], axis=1)

del train_df

X_train=X_train.values
Y_train=Y_train.values
test_final=test_final.values


X_train=X_train.reshape(42000,28,28,1)
test_final=test_final.reshape(28000,28,28,1)

X_train = X_train.astype('float32')
print("Size of original train data set=", X_train.shape[0])

#===============================================================================
# 			Data Augmentation -1
#===============================================================================

def augment_data(dataset, dataset_labels, augementation_factor=1, 
					  use_random_rotation=True, 
					  use_random_shear=True, 
					  use_random_shift=True, 			
					  use_random_zoom=True):
	augmented_image = []
	augmented_image_labels = []

	for num in range (0, dataset.shape[0]):

		for i in range(0, augementation_factor):
			# original image:
			augmented_image.append(dataset[num])
			augmented_image_labels.append(dataset_labels[num])

			if use_random_rotation:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[num], 20, row_axis=0, 								col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shear:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num], 0.2, row_axis=0, 							col_axis=1,  channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shift:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, 								col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_zoom:
				augmented_image.append(tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], (0.9,0.9), row_axis=0, 								col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

	return np.array(augmented_image), np.array(augmented_image_labels)


#===============================================================================
# 			Data Augmentation -2: Feature Standardization, ZCA whitening
#===============================================================================

X_train_aug2=[]
Y_train_aug2=[]

### Feature Standardization
datagen = ImageDataGenerator(featurewise_center=True, 
			     featurewise_std_normalization=True)
			     
datagen.fit(X_train)
X_train_aug1, Y_train_aug1 = datagen.flow(X_train, Y_train, batch_size=42000).next()


## ZCA Whitening
datagen2 = ImageDataGenerator(zca_whitening=True)
datagen2.fit(X_train)

X_train_aug2, Y_train_aug2 = datagen2.flow(X_train, Y_train, batch_size=42000).next()	

#===============================================================================
# 			Combine all the Data Augmentation
#===============================================================================

print("Size of original train data set=", X_train.shape[0])
X_train,Y_train= augment_data(dataset=X_train, dataset_labels=Y_train)
print("Size of Augmented train data set before Aug1=", X_train.shape[0])

X_train = np.concatenate((X_train, X_train_aug1))
Y_train = np.concatenate((Y_train, Y_train_aug1))

X_train = np.concatenate((X_train, X_train_aug2))
Y_train = np.concatenate((Y_train, Y_train_aug2))

print("Size of Augmented train data set Aug2=", X_train.shape[0])


#===============================================================================
# 			Normalize the data
#===============================================================================
X_train = X_train/255.0

#===============================================================================
# 			Test train split
#===============================================================================
(trainData, testData, trainLabels, testLabels) = train_test_split(
	X_train, Y_train, test_size=0.1)

#===============================================================================
# 			COnvert labels to categorical vector
#===============================================================================
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

#===============================================================================
# 			Prepare the NN model
#===============================================================================

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(numChannels=1, imgRows=28, imgCols=28,
		    numClasses=10,
		    weightsPath=args["weights"] if args["load_model"] > 0 else None)
		    
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

#===============================================================================
# 			Train the model
#===============================================================================

#Train the model if the pre-trained model is not available
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(trainData, trainLabels, batch_size=128, epochs=2,
		verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)
################################################################################

#===============================================================================
# 			Test time augmentation of the data
#===============================================================================

def test_augment_data(dataset, augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, 				use_random_zoom=True):
	augmented_image = []
	#augmented_image_labels = []

	for i in range(0, augementation_factor):
		# original image:
		augmented_image.append(dataset)
		#augmented_image_labels.append(dataset_labels[num])

		if use_random_rotation:
			augmented_image.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset, 20, row_axis=0, 							col_axis=1, channel_axis=2))
			#augmented_image_labels.append(dataset_labels[num])

		if use_random_shear:
			augmented_image.append(tf.contrib.keras.preprocessing.image.random_shear(dataset, 0.2, row_axis=0, 						col_axis=1,  channel_axis=2))
			#augmented_image_labels.append(dataset_labels[num])

		if use_random_shift:
			augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset, 0.2, 0.2, row_axis=0, 							col_axis=1, channel_axis=2))
			#augmented_image_labels.append(dataset_labels[num])

		if use_random_zoom:
			augmented_image.append(tf.contrib.keras.preprocessing.image.random_zoom(dataset, (0.9,0.9), row_axis=0, 							col_axis=1, channel_axis=2))
			#augmented_image_labels.append(dataset_labels[num])

	return np.array(augmented_image)

#===============================================================================
# 			Evaluate the model on test data
#===============================================================================

predictions=[]
for num in range(0, test_final.shape[0]):
	if (num % 1000)==0:
		print("Testimage number:",num)
	single_test_image= test_augment_data(dataset=test_final[num])
	single_test_image= single_test_image/255
	pred = model.predict(single_test_image)
	pred_mean= pred.mean(axis=0)
	predictions.append(pred_mean)

test_final_predictions=(np.array(predictions)).argmax(axis=1)	
np.savetxt('test_final_predictions_test_aug_method.csv', test_final_predictions, delimiter=",")

#===============================================================================
# 			Visualize some of the saple predictions
#===============================================================================
# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
	# classify the digit
	
	probs = model.predict(testData[np.newaxis, 20471])
	prediction = probs.argmax(axis=1)

	# extract the image from the testData if using "channels_first"
	# ordering
	if K.image_data_format() == "channels_first":
		image = (testData[i][0] * 255).astype("uint8")

	# otherwise we are using "channels_last" ordering
	else:
		image = (testData[i] * 255).astype("uint8")

	# merge the channels into one image
	image = cv2.merge([image] * 3)

	# resize the image from a 28 x 28 image to a 96 x 96 image so we
	# can better see it
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)

	# show the image and prediction
	cv2.putText(image, str(prediction[0]), (5, 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
		np.argmax(testLabels[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
#===============================================================================
# 				Done
#===============================================================================
