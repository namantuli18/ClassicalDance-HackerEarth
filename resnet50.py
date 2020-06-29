import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen,urlretrieve
from PIL import Image
from tqdm import tqdm_notebook
from sklearn.utils import shuffle
import cv2
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
EARLY_STOP_PATIENCE = 3

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#from resnets_utils import *
from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers
import tensorflow as tf
IMG_SIZE=(64,64)
image_size=64
#from tf_utils import convert_to_one_hot
from data import create_train_data
train_x,train_y,test_x,test_y=create_train_data()
train_x = train_x/255
test_x = test_x/255
CHANNELS = 3

print(train_x.shape,train_y.shape)

'''IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100


print(train_y)
num_classes = 8
base_model = applications.resnet50.ResNet50(weights= 'imagenet', include_top=False, input_shape= (64,64,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
from keras.optimizers import SGD, Adam
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
dataset_mod=["bharatanatyam",'kathak', 'kathakali', 'kuchipudi', 'manipuri', 'mohiniyattam', 'odissi', 'sattriya']
train_generator = data_generator.flow_from_directory(
        'dataset/train',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'dataset/test',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical')



print(train_generator.class_indices)
fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs = NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
       # callbacks=[cb_checkpointer, cb_early_stopper]
)

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.01)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs = 10, batch_size = 64)
FINAL_PATH='G:/resume/ML_Hackathons/Dance Form/d/dataset/test'

model = Sequential()

# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(ResNet50(include_top = False, pooling = 'avg', weights = None))

# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
model.add(Dense(8, activation = 'softmax'))
model.summary()

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False
sgd = optimizers.SGD(lr = 1e-3, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

'''
'''
FINAL_PATH=r'G:\resume\ML_Hackathons\Dance Form\d\dataset\test'

def write__():
  with open('submission250.csv',"w") as f:
    f.write('Image,target\n')
  for files in os.listdir(FINAL_PATH):
    with open('submission250.csv',"a" ) as f:

      path=os.path.join(FINAL_PATH,files)

      image=cv2.imread(path)
      image=cv2.resize(image,(300,300))
      image = np.array(image)
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      #image = preprocess_input(image)
      yhat = model.predict(image)
      #print(yhat)
      #label = decode_predictions(yhat,top=8)
      #print(files,dataset_mod[np.argmax(yhat)])
      label=yhat
      #print(yhat)
      str_=[np.argmax(yhat)][0]
      print(str_)
      label = np.argmax(label)
                        #f.write(f'{files},{dataset_mod[np.argmax(yhat)]}\n')
                        print(files,dataset_mod[label])
                        f.write(f'{files},{dataset_mod[label]}\n')

      f.write(f'{files},{dataset_mod[str_]}\n')
write__()'''