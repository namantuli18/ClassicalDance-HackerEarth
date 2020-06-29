import tensorflow as tf
from keras.models import Model
from keras.applications import MobileNetV2, ResNet50, InceptionV3 # try to use them and see which is better
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import get_file
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import os
import pathlib
import numpy as np
import cv2
TRAIN_DIR='dataset/train'
TEST_DIR='dataset/test'
WIDTH = 224
HEIGHT = 224
BATCH_SIZE = batch_size= 32

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
dataset=['bharatanatyam',"kathak",'kathakali','kuchipudi','manipuri','mohiniyattam','odissi','sattriya']
print(train_generator.class_indices)
def write__():

  FINAL_PATH=r'G:\resume\ML_Hackathons\Dance Form\d\dataset\test'
  with open('1submission250.csv',"w") as f:
    f.write('Image,target\n')
  for files in os.listdir(FINAL_PATH):
    with open('1submission250.csv',"a" ) as f:

      path=os.path.join(FINAL_PATH,files)

      image=cv2.imread(path)
      image=cv2.resize(image,(224,224))
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
      print(files,str_)
      label = np.argmax(label)

                        #f.write(f'{files},{dataset_mod[np.argmax(yhat)]}\n')
      #print(files,dataset[label])
                        #f.write(f'{files},{dataset_mod[label]}\n')

      f.write(f'{files},{dataset[str_]}\n')


def create_model(input_shape):
    num_classes=8
    # load MobileNetV2
    model = MobileNetV2(input_shape=input_shape,weights='imagenet')
    # remove the last fully connected layer
    #model.layers.pop()
    # freeze all the weights of the model except the last 4 layers
    for layer in model.layers[:-4]:
        layer.trainable = False
    # construct our own fully connected layer for classification
    output = Dense(num_classes, activation="softmax")
    # connect that dense layer to the model
    output = output(model.layers[-1].output)
    model = Model(inputs=model.inputs, outputs=output)
    # print the summary of the model architecture
    model.summary()
    # training the model using rmsprop optimizer
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # load the data generators
    IMAGE_SHAPE=(224,224,3)
    # constructs the model
    model = create_model(input_shape=IMAGE_SHAPE)
    # model name
    model_name = "MobileNetV2_finetune_last5"
    # some nice callbacks
    tensorboard = TensorBoard(log_dir=f"logs/{model_name}")
    '''checkpoint = ModelCheckpoint(f"results/{model_name}" + "-loss-{val_loss:.2f}-acc-{val_acc:.2f}.h5",
                                            save_best_only=True,
                                            verbose=1)'''
    # make sure results folder exist
    if not os.path.isdir("results"):
        os.mkdir("results")
    # count number of steps per epoch
    training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
    validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)
    # train using the generators
    model.fit_generator(train_generator, steps_per_epoch=training_steps_per_epoch,
                        
                        epochs=5, verbose=1)
    write__()
