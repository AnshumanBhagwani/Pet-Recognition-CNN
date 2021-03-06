from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#Initialization
classifier = Sequential()

#convolution
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))
#Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#second layer
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Third layer
classifier.add(Convolution2D(64,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#Flattening
classifier.add(Flatten())


#Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compilation
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



from keras.preprocessing.image import ImageDataGenerator 


#fitting CNN to the images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

trset = train_datagen.flow_from_directory(
                                            'dataset/training_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

tstset = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        trset,
        steps_per_epoch=250,
        epochs=25,
        validation_data=tstset,
        validation_steps=62)


#to check for single3 test image
'''
import numpy as np

from keras.preprocessing import image

tstimg1 = image.load_img('dataset/single_prediction/qw.jpg', target_size = (64, 64))
tstimg1 = image.img_to_array(tstimg1)
tstimg1 = np.expand_dims(tstimg1 , axis = 0)
result1 = classifier.predict(tstimg1)
trset.class_indices
'''


'''from keras import backend as K
K.tensorflow_backend._get_available_gpus()


import tensorflow as tf
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))'''
