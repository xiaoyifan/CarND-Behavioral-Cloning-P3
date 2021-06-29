import csv
import cv2
import numpy as np
import scipy
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Activation, MaxPooling2D, Dropout, Lambda, Cropping2D

def resize(image, new_dim):
	return cv2.resize(image, new_dim, interpolation = cv2.INTER_AREA)

def crop(image, top_pixels, bottom_pixel):
    top = top_pixels
#     print(type(image))
    bottom = image.shape[0] - bottom_pixel

    return image[top:bottom, :]

def preprocess_image(image_path):
    center_name = image_path.split('/')[-1]
    current_image_path = './data/IMG/' + center_name
    img = cv2.imread(current_image_path)
#     print("path: ", type(img))
    cropped_image = crop(img, 50, 20)
    resized_image = resize(cropped_image, (64, 64))
    img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    img = (img / 255.0) - 0.5
    return img


car_images = []
steering_angles = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	first_line = True
	for row in reader:
		if first_line:
			first_line = False   
		else:	
			steering_center = float(row[3])

			# create adjusted steering measurements for the side camera images
			correction = 0.2 # this is a parameter to tune
			steering_left = steering_center + correction
			steering_right = steering_center - correction

			# read in images from center, left and right cameras
			img_center = preprocess_image(row[0])

			img_left = preprocess_image(row[1])

			img_right = preprocess_image(row[2])

			# add images and angles to data set
			car_images.extend([img_center, img_left, img_right, np.fliplr(img_center), np.fliplr(img_left), np.fliplr(img_right)])
			steering_angles.extend([steering_center, steering_left, steering_right, -1*steering_center, -1*steering_left, -1*steering_right])


x_train = np.array(car_images)
y_train = np.array(steering_angles)
print("y_train_count: ", len(y_train))
print("y_train: ", y_train)

model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((50,20), (0,0))))
# model.add(Resizing(64, 64))

model.add(Conv2D(24, (5, 5), padding='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(36, (5, 5), padding='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(48, (5, 5), padding='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64, (3, 3), padding='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64, (3, 3), padding='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dropout(0.6))

model.add(Dense(100))
model.add(Activation('relu'))
# model.add(Dropout(0.6))

model.add(Dense(50))
model.add(Activation('relu'))
# model.add(Dropout(0.6))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001))
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, epochs=6)

model.save('model_1.h5')