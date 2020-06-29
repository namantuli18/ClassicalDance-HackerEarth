import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import tflearn
from tflearn.layers.conv import max_pool_2d,conv_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from random import shuffle

TRAIN_DIR='d/dataset/train'
TEST_DIR='d/dataset/test'

IMG_SIZE=214

dataset=['manipuri','bharatanatyam','odissi','kathakali','kathak','sattriya','kuchipudi','mohiniyattam']

LR=1e-3
def return_class(img):
	df=pd.read_csv('dataset/train.csv')
	ret_=np.zeros(8,int)
	#print(ret_)
	for count,i in enumerate(df.Image):
		if str(i) == img:
			str_=(df.target[count])
			
			for counter,i in enumerate(dataset):
				if i==str_:
					#print(str_,count)
					ret_[counter]+=1
					#print(ret_)
					return str_

					
def create_train_data():
	training_data=[]
	train_x=[]
	train_y=[]
	test_x=[]
	test_y=[]
	df=pd.read_csv('dataset/train.csv')
	for file in tqdm(os.listdir(TRAIN_DIR)):
		path=os.path.join(TRAIN_DIR,file)
		label=return_class(file)
		#print(label)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR),(192,192))
		#print(img.shape)
		#cv2.imshow('img',img)
		training_data.append([np.array(img),np.array(label)])
		#print(img.shape)
		shuffle(training_data)
		train_x.append(np.array(img))
		train_y.append(label)
		#print(train_y)
	for file in tqdm(os.listdir(TRAIN_DIR)):
		path=os.path.join(TRAIN_DIR,file)
		label=return_class(file)
			#print(label)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR),(192,192))
			#print(img.shape)
			#cv2.imshow('img',img)
		training_data.append([np.array(img),np.array(label)])
			#print(img.shape)
		shuffle(training_data)
		test_x.append(np.array(img))
		test_y.append(label)
		break


		#print(np.array(train_y).shape)
	return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)
#train_x,train_y,test_x,test_y=create_train_data()
#print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

'''create_train_data()
'''


def create_test_data():
	testing_data=[]
	
	for file in tqdm(os.listdir(TEST_DIR)):
		path=os.path.join(TEST_DIR,file)
		
		img=cv2.resize(cv2.imread(path),(IMG_SIZE,IMG_SIZE))
		
		testing_data.append([np.array(img),np.array(file)])
		
	return testing_data
'''train_data=create_train_data()
test_data=create_test_data()

convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')

convnet = conv_2d(convnet, 64, 8, activation='relu')
convnet = max_pool_2d(convnet, 8)

convnet = conv_2d(convnet, 128, 8, activation='relu')
convnet = max_pool_2d(convnet, 8)

convnet = conv_2d(convnet, 256, 8, activation='relu')
convnet = max_pool_2d(convnet, 8)

convnet = conv_2d(convnet, 512, 8, activation='relu')
convnet = max_pool_2d(convnet, 8)

convnet = conv_2d(convnet, 256, 8, activation='relu')
convnet = max_pool_2d(convnet, 8)

convnet = conv_2d(convnet, 128, 8, activation='relu')
convnet = max_pool_2d(convnet, 8)

convnet = conv_2d(convnet, 64, 8, activation='relu')
convnet = max_pool_2d(convnet, 8)


convnet=fully_connected(convnet,1024,activation='relu')
convnet=dropout(convnet,0.8)


convnet = fully_connected(convnet, 8, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')
train=train_data[:-1]
test=train_data[-1:]
X=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)/255
Y=[i[1] for i in train]

test_x=np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)/255

test_y=[i[1] for i in test]'''
#print(test_x,test_y)

'''model.fit({'input': X}, {'targets': Y}, n_epoch=100, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id='MODEL_NAME' )
with open('submission3.csv','w') as f:
	f.write('Image,target\n')
fig=plt.figure()
for data in test_data:
	with open('submission3.csv','a') as f:
		img_num=data[1]
		img_data=data[0]
		

			#y=fig.add_subplot(3,4,num+1)
		orig=img_data
		data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
		model_out=model.predict([data])[0]
		counter=np.argmax(model_out)
		str__=dataset[counter]
		f.write('{},{}\n'.format(img_num,str__))'''