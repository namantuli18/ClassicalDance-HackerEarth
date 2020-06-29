import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen,urlretrieve
from PIL import Image
from tqdm import tqdm_notebook

from sklearn.utils import shuffle
import cv2
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
from data import create_train_data
train_x,train_yo,test_x,test_y=create_train_data()
train_x = train_x/255
test_x = test_x/255
import numpy as np
dataset=['manipuri','bharatanatyam','odissi','kathakali','kathak','sattriya','kuchipudi','mohiniyattam']
def train__y():
	lis=[]
	for i in train_yo:
		a=np.zeros(8,int)
		for count,j in enumerate(dataset):
			if i==j:
				a[count]+=1
				break
		lis.append(a.tolist())

	#print(lis)
	return np.array(lis)
train_y=train__y()


print(train_x.shape,train_y.shape)
img_height,img_width = 64,64 
num_classes = 8
#If imagenet weights are being loaded, 
#input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
base_model = applications.resnet50.ResNet50(weights= 'imagenet', include_top=False, input_shape= (192,192,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
from keras.optimizers import SGD, Adam
sgd = SGD(lr=1e-3, momentum=0.9, decay=0.9, nesterov=False)
adam = Adam(lr=1e-4)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x,train_y,epochs=10,batch_size=64)

model.summary()

def write__():
  FINAL_PATH=r'G:\resume\ML_Hackathons\Dance Form\d\dataset\test'
  with open('1submission250.csv',"w") as f:
    f.write('Image,target\n')
  for files in os.listdir(FINAL_PATH):
    with open('1submission250.csv',"a" ) as f:

      path=os.path.join(FINAL_PATH,files)

      image=cv2.imread(path)
      image=cv2.resize(image,(192,192))
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