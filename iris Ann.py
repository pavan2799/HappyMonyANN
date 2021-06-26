# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 19:51:10 2021

@author: hp
"""

import pandas as pd
import numpy as np

iris = pd.read_csv("C:/Users/hp/OneDrive/Desktop/HR/Iris.csv")
iris.describe()
iris.columns

iris = iris.drop(columns = 'Id') #as it is discrete it doesnt give much information.

#converting the catergorical variable to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
iris.Species = le.fit_transform(iris.Species)

iris.isna().sum()  # we check for any null values

import matplotlib.pyplot as plt

SepalLengthCm = plt.boxplot(iris.SepalLengthCm)
plt.title('SepalLengthCm boxplot')

SepalWidthCm = plt.boxplot(iris.SepalWidthCm)
plt.title('SepalWidthCm boxplot')

#Removal outliers in sepalwidthcm
iqr=iris.SepalWidthCm.quantile(0.75)-iris.SepalWidthCm.quantile(0.25)
lower_limit=iris.SepalWidthCm.quantile(0.25)-(iqr*1.5)
upper_limit=iris.SepalWidthCm.quantile(0.75)+(iqr*1.5)
outlier=np.where(iris.SepalWidthCm > upper_limit, True, np.where(iris.SepalWidthCm < lower_limit, True, False))
iris_trimmed = iris.loc[~(outlier)]
iris.shape
iris_trimmed.shape
trimmed_boxplot = plt.boxplot(iris_trimmed.SepalWidthCm)
plt.title('removal SepalWidthCm outliers')
plt.show()

PetalLengthCm = plt.boxplot(iris.PetalLengthCm)
plt.title('PetalLengthCm boxplot')

PetalWidthCm = plt.boxplot(iris.PetalWidthCm)
plt.title('PetalWidthCm boxplot')

Species = plt.boxplot(iris.Species)
plt.title('Species boxplot')

from sklearn.feature_selection import VarianceThreshold
var_threshold = VarianceThreshold(threshold=0) 
data_var=var_threshold.fit(iris) 
# We can check the variance of different features as
print(var_threshold.variances_)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.utils import np_utils

input = iris.iloc[: , 0:4]
target = iris.iloc[: , 4]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input , target , test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)

model = Sequential() #we are building the model
model.add(Dense(4, input_shape=(4,), activation='linear',)) # we have used linear activation function(k0 + k1x) 
model.add(Dense(4,activation='tanh'))
model.add(Dense(3,activation='softmax',)) #we have used output layer with softmax activation function
"""
Sgd
"""
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy']) # here we have used catergorical cross entropy
hist = model.fit(x=x_train,y=y_train,batch_size=10,epochs=100,validation_data=(x_val, y_val))

eval_score_test = model.evaluate(x_test,y_test,verbose = 0)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 

eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100))


loss_train = hist.history['loss']
loss_val = hist.history['val_loss']
epochs = range(100)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss_train = hist.history['accuracy']
loss_val = hist.history['val_accuracy']
epochs = range(100)
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
f1_score = classification_report(y_test_class, y_ped_class)
confusion_matrix(y_test_class, y_ped_class)

y_pred = model.predict(x_train)
y_train_class = np.argmax(y_train, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
f1_score_train = classification_report(y_train_class, y_pred_class)
confusion_matrix(y_train_class, y_pred_class)


hist.history

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(["train"], loc='upper left')
plt.show()


plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(["train"], loc='upper left')
plt.show()


############################################ADDITIONAL ##############################################


"""
adadelta
"""
model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train,y=y_train,batch_size=10,epochs=100)

eval_score_test = model.evaluate(x_test,y_test,verbose = 0)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 

eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100))
"""
Adagrad
"""
model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train,y=y_train,batch_size=10,epochs=100)

eval_score_test = model.evaluate(x_test,y_test,verbose = 0)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 

eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100))
"""
Adamax
"""
model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train,y=y_train,batch_size=10,epochs=100)

eval_score_test = model.evaluate(x_test,y_test,verbose = 0)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 

eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100))
"""
adam
"""
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x=x_train,y=y_train,batch_size=10,epochs=100)

eval_score_test = model.evaluate(x_test,y_test,verbose = 0)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 

eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100))
"""
RMSprop
"""
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x=x_train,y=y_train,batch_size=10,epochs=100)


eval_score_test = model.evaluate(x_test,y_test,verbose = 0)
print ("Accuracy: %.3f%%" %(eval_score_test[1]*100)) 

eval_score_train = model.evaluate(x_train,y_train,verbose=0)
print ("Accuracy: %.3f%%" %(eval_score_train[1]*100))

y_pred = model.predict(x_test)
y_test_class = np.argmax(y_test, axis=1)
y_ped_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
f1_score = classification_report(y_test_class, y_ped_class)
confusion_matrix(y_test_class, y_ped_class)
accuracy = (11+4+6)/(30)

y_pred = model.predict(x_train)
y_train_class = np.argmax(y_train, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
classification_report(y_train_class, y_pred_class)
confusion_matrix(y_train_class, y_pred_class)
accuracy = (39+20+44)/(120)

hist.history

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(["train"], loc='upper left')
plt.show()


plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(["train"], loc='upper left')
plt.show()



