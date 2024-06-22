import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from keras import utils
import keras.api.utils
import keras.api.models
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.api.metrics import Precision, Recall, BinaryAccuracy
from keras.api.models import load_model


"""
The .what() method of the imghdr module in Python is used to determine the type of 
image contained in a file or a byte stream based on the first few bytes of the data. 
It takes a single argument, which can be: (eg. jpeg, png, svg)

File Path: A string representing the path to the image file on your system.
Byte Stream: A bytes-like object containing the image data itself.
"""

# where my images are located
data_dir = 'data'
# the allowed image extensions
img_exts = ['jpeg', 'jpg', 'png', 'bmp']

# Remove dodgy images

# loop through the folders in the directory and go through each image in each folder. A folder is an image class, i.e a label.
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            cv2.imread(image_path)
            ext = imghdr.what(image_path)
            # if the extension is invalid then the file is deleted.
            if ext not in img_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
            else:
                continue
        except Exception as e:
            print('Issue with image {}'.format(image_path))

# Load the data, but not into the memory.

data = keras.utils.image_dataset_from_directory('data')

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
    
# Scaling data

data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

# Splitting data into sets

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#train your model.
train
print(train)

#Creating a CNN with layers for the sequential model
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

#basic logging for epochs
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

#plot the losses using matplotlib
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#plot the accuracy using matplotlib
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#evaluation segment
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result(), re.result(), acc.result())

#test
img = cv2.imread('depositphotos_391577690-stock-photo-cheerful-woman-outstretched-hands-sitting.jpeg')
plt.imshow(img)
plt.show()
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
yhat = model.predict(np.expand_dims(resize/255, 0))
if yhat > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')
    
#save the model
model.save(os.path.join('models','imageclassifier.h5'))
