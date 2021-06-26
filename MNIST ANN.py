# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 16:03:23 2021

@author: hp
"""

import pandas as pd
import numpy as np

mnist_train = pd.read_csv("C:/Users/hp/OneDrive/Desktop/HR/mnist_train.csv")
mnist_test = pd.read_csv("C:/Users/hp/OneDrive/Desktop/HR/mnist_test.csv")

import numpy as np
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout,Flatten
from keras.utils import np_utils
from keras.optimizers import SGD

x_train = mnist_train.iloc[:,1:].values.astype("float32")
x_test = mnist_test.iloc[:,1:].values.astype("float32")
y_train = mnist_train.label.values.astype("float32")
y_test = mnist_test.label.values.astype("float32")

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)


# Normalizing the inputs to fall under 0-1 by 
# diving the entire data with 255 (max pixel value)
x_train = x_train/255
x_test = x_test/255
x_val = x_val/255

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(150,input_dim =784,activation="relu"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="SGD",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
hist = model.fit(x=x_train,y=y_train,batch_size=1000,epochs=20 , validation_data=(x_val, y_val))

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 0)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set

hist.history

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(["train"], loc='upper left')
plt.show()


plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(["train"], loc='upper left')
plt.show()

loss_train = hist.history['loss']
loss_val = hist.history['val_loss']
epochs = range(20)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss_train = hist.history['accuracy']
loss_val = hist.history['val_accuracy']
epochs = range(20)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='Test accuracy')
plt.title('Training and Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_pred = model.predict(x_test)
y_test_class = np.argmax(y_test, axis=1)
y_ped_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
f1_score_test = classification_report(y_test_class, y_ped_class)
confusion_matrix(y_test_class, y_ped_class)


y_pred = model.predict(x_train)
y_train_class = np.argmax(y_train, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
f1_score_train = classification_report(y_train_class, y_pred_class)
confusion_matrix(y_train_class, y_pred_class)





"""
MNIST
"""

import pandas as pd
import numpy as np

mnist_train = pd.read_csv("C:/Users/hp/OneDrive/Desktop/HR/mnist_train.csv")
mnist_test = pd.read_csv("C:/Users/hp/OneDrive/Desktop/HR/mnist_test.csv")

import numpy as np
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout,Flatten
from keras.utils import np_utils

x_train = mnist_train.iloc[:,1:].values.astype("float32")
x_test = mnist_test.iloc[:,1:].values.astype("float32")
y_train = mnist_train.label.values.astype("float32")
y_test = mnist_test.label.values.astype("float32")

# Normalizing the inputs to fall under 0-1 by 
# diving the entire data with 255 (max pixel value)
x_train = x_train/255
x_test = x_test/255

# one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape

def design_mlp():
    # Initializing the model 
    model = Sequential()
    model.add(Dense(150,input_dim =784,activation="relu"))
    model.add(Dense(200,activation="tanh"))
    model.add(Dense(num_of_classes,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

# building a cnn model using train data set and validating on test data set
model = design_mlp()

# fitting model on train data
hist = model.fit(x=x_train,y=y_train,batch_size=1000,epochs=20)

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 
# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100)) 
# accuracy on train data set

hist.history

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(["train"], loc='upper left')
plt.show()


plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(["train"], loc='upper left')
plt.show()

y_pred = model.predict(x_test)
y_test_class = np.argmax(y_test, axis=1)
y_ped_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
classification_report(y_test_class, y_ped_class)
confusion_matrix(y_test_class, y_ped_class)


y_pred = model.predict(x_train)
y_train_class = np.argmax(y_train, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
classification_report(y_train_class, y_pred_class)
confusion_matrix(y_train_class, y_pred_class)