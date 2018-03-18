-*- coding: utf-8 -*-

import pandas as pd
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score

#Read Data
data2 = pd.read_csv('C:/Users/surya/Desktop/SpringSemester/IDA/HW1/Dataset/Biomechanical_Data_column_2C_weka.csv')

#The dataset contains 310 records
#See the Shape of data and check the first 5 records of your data
print("Data2 Shape: \n", data2.shape)
print("Data2 Head: \n", data2.head())

#Shuffling data before training the dataset
shuffledata2 = shuffle(data2)

#Print the first 5 records of the shuffled data
print("Data2Shuffle head: \n", shuffledata2.head())

#Assign the first 210 records to the Training data
train_data2 = shuffledata2[:210]
#Assign the last 100 records to the Testing data
test_data2 = shuffledata2[-100:]

#Check the shape and head of the training and testing data
print("Train Data2 Shape: \n", train_data2.shape)
print("Train Data2 head: \n", train_data2.head())
print("Test Data2 Shape: \n", test_data2.shape)
print("Test Data2 head: \n", test_data2.head())

#Assign the first 6 column names as the Train Features
trainFeaturesData = list(train_data2.columns[:6])
#print("Features: \n", trainFeaturesData)

#Get the data under these 6 columns
trainFeatures = train_data2[trainFeaturesData]
#print("Train Features: \n", trainFeatures.head())

#Get the data under the 'class' column
trainClass = train_data2["class"]
#print("Training data Class: \n", trainClass.head())

#Create a SVM Classifier
#classifier = svm.SVC(kernel='rbf', C=1, gamma=0.001)
#classifier = svm.SVC(kernel='rbf', C=1, gamma=0.167) #auto
classifier = svm.SVC(kernel='rbf', C=2.2, gamma=0.001)
#classifier = svm.SVC(kernel='rbf')

#Using the Training data to build the classifier
classifier.fit(trainFeatures, trainClass)
print("SVM Classifier: \n", classifier)

# get support vectors
supportVectors = classifier.support_vectors_
print("Support Vectors: \n", supportVectors)

# get indices of support vectors
supportVectorIndices = classifier.support_
print("Support Vector Indices: \n", supportVectorIndices)
