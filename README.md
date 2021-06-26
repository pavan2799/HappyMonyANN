# HappyMonyANN
ANN for irsis and MNIST dataset

Documentation for ANN on iris dataset: -
Business objective: - Design a Neural Network on IRIS data where the activation function learned from the data.
Constrain: - Collection of data, Building the model with 2 hidden layers, using catergorical loss entropy.
Data pre-processing: - We drop the Id column as it is discrete and it doesn’t give much information.
We convert the categorical column into numeric by using Label Encoder on Species which is the target feature.
We check for any null values, missing numbers and there are no missing values and null values in our data.
We check for any outliers, which will effect the dataset while we build the model.
We trim our outliers.
   
  
Model Building: -
We import necessary packages and we build the model.
Using train test split we split the model in which we split the data on test is 20% and train 80%.
We convert ytrain and ytest to (np_utils.to_) categorical is used to convert array of labeled data (from 0 to nb_classes - 1) to one-hot vector.
We build the sequential model which helps us to take make layers.
We build the input layer using activation = ‘linear’ which helps to solve y = k0 + k1x.
We build the next layer using activation function = ‘tanh’
We build the output layer using activation function = ‘Softmax’ which is use in multi class classification.
We fit the using optimizer = SGD then we can see results
Accuracy test: - 0.93
Accuracy train: - 0.97
F1 score test : -
Precision 	recall 		f1-score 	support

0 		1.00 		1.00 		1.00 		11
1 		1.00 		0.85 		0.92 		13
2 		0.75 		1.00 		0.86 		6

accuracy 					0.93 		30
macro avg 	0.92 		0.95 		0.92 		30
weighted avg 	0.95 		0.93 		0.94 		30

F1 score train: -
precision 	recall 		f1-score 	support

0 		1.00 		1.00 		1.00 		31
1 		1.00 		0.90 		0.95 		30
2 		0.92 		1.00 		0.96 		35

accuracy 					0.97 		96
macro avg 	0.97 		0.97 		0.97 		96
weighted avg 	0.97 		0.97 		0.97 		96

Loss function vs Epochs: -
 



Training vs Test loss: -
 

Training vs Test Accuracy: -
 
Model 	Test accuracy	Train accuracy
Adam	66.6	80
Ada delta	20	36.66
Ada grad	0.833	46.667
Ada max	63.33	75.5
RMSprop	100	95.833
 
I have also done done for MNIST dataset.
