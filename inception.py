from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
base_model = InceptionV3(weights='imagenet', include_top=False)
CLASSES = 8
    
# setup model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
   
# transfer learning
for layer in base_model.layers:
    layer.trainable = False
      
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR='dataset/train'
TEST_DIR='dataset/test'



WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 32

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
		batch_size=BATCH_SIZE,
		class_mode='categorical')
    
validation_generator = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')
print(train_generator.class_indices)
EPOCHS = 35
BATCH_SIZE = 32
STEPS_PER_EPOCH = 12
VALIDATION_STEPS = 12

MODEL_FILE = 'filename.model'

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)
  
model.save(MODEL_FILE)

dataset=['bharatanatyam',"kathak",'kathakali','kuchipudi','manipuri','mohiniyattam','odissi','sattriya']

def write__():
  FINAL_PATH=r'G:\resume\ML_Hackathons\Dance Form\d\dataset\test'
  with open('1submission250.csv',"w") as f:
    f.write('Image,target\n')
  for files in os.listdir(FINAL_PATH):
    with open('1submission250.csv',"a" ) as f:

      path=os.path.join(FINAL_PATH,files)

      image=cv2.imread(path)
      image=cv2.resize(image,(299,299))
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
      print(files,dataset[label])
                        #f.write(f'{files},{dataset_mod[label]}\n')

      f.write(f'{files},{dataset[str_]}\n')
write__()