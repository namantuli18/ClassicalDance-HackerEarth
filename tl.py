from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pickle
import os

# re-size all the images to this
IMAGE_SIZE = [400, 400]

train_path = 'dataset/train'
valid_path = 'dataset/test'
dataset=['manipuri','bharatanatyam','odissi','kathakali','kathak','sattriya','kuchipudi','mohiniyattam']
           #kathak    #'manipuri'    #kuchipudi  #odissi     #bharat   #mohin    #kathakali     #sattriya
dataset_mod=["bharatanatyam",'kathak', 'kathakali', 'kuchipudi', 'manipuri', 'mohiniyattam', 'odissi', 'sattriya']

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False,classes=8)
model = Sequential()

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  
vgg.layers.pop()
  
  # useful for getting number of classes
folders = glob('dataset/train/*')
  

# our layers - you can add more if you want
x = Flatten()(vgg.output)
x = Dense(312, activation='relu')(x)
#x = Dense(8, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True


)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (400, 400),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

'''test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size = (248, 248),
                                            batch_size = 32,
                                            class_mode = 'categorical')'''
print(training_set.class_indices)
'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model

#print(test_set.labels)
r = model.fit_generator(
        training_set,
        #validation_data=test_set,
        epochs=10,
        steps_per_epoch=len(training_set),
        validation_steps=len(training_set)
      )
# loss
'''plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
#plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
#plt.show()
plt.savefig('AccVal_acc')'''

import tensorflow as tf

from keras.models import load_model

import cv2
'''with open('model.pickle',"rb") as f:
  model=pickle.load(f)
'''


FINAL_PATH='FINAL'
def write__():
  with open('submission.csv',"w") as f:
    f.write('Image,target\n')
  for files in os.listdir(FINAL_PATH):
    with open('submission.csv',"a" ) as f:

      path=os.path.join(FINAL_PATH,files)

      image=cv2.imread(path)
      image=cv2.resize(image,(400,400))
      image = np.array(image)
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      image = preprocess_input(image)
      yhat = model.predict(image)
      #print(yhat)
      #label = decode_predictions(yhat,top=8)
      #print(files,dataset_mod[np.argmax(yhat)])
      label=yhat
      str_=dataset_mod[np.argmax(yhat)]
      '''label = np.argmax(label)
                        #f.write(f'{files},{dataset_mod[np.argmax(yhat)]}\n')
                        print(files,dataset_mod[label])'''
      #f.write(f'{files},{dataset_mod[label]}\n')

      f.write(f'{files},{str_}\n')
write__()


#label = label[0][0]

#print('%s (%.2f%%)' % (label[1], label[2]*100))