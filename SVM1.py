#-*- coding: utf-8 -*-

import pandas as pd
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score

#Read Data
data2 = pd.read_csv('C:/Users/surya/Desktop/SpringSemester/IDA/Dataset/Biomechanical_Data_column_2C_weka.csv')

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
#For a Linear Kernel, only C value must be optimised. There is no need to optimise the gamma parameter. Gamma is only used for rbf kernels
#classifier = svm.SVC(kernel='linear')
#classifier = svm.SVC(kernel='linear', C=2.2, gamma=0.001)
classifier = svm.SVC(kernel='linear', C=2.2)

#Using the Training data to build the classifier
classifier.fit(trainFeatures, trainClass)
print("SVM Classifier: \n", classifier)

# get support vectors
supportVectors = classifier.support_vectors_
print("Support Vectors: \n", supportVectors)

# get indices of support vectors
supportVectorIndices = classifier.support_
print("Support Vector Indices: \n", supportVectorIndices)

"""#print(type(supportVectorIndices))
supportVectorIndicesdf = pd.DataFrame(supportVectorIndices)
print("**********************")
#print(train_data2.loc[3,:])

for i in range(0, len(supportVectorIndicesdf)):
    #print(supportVectorIndicesdf[i])
    print(train_data2.loc[supportVectorIndicesdf[i],:])
    #print(train_data2[supportVectorIndices[i]])
print("**********************")"""

# get number of support vectors for each class - In Ascending Order
supportVectorCount = classifier.n_support_
print("Support Vectors Count: \n", supportVectorCount)
print("Support Vectors Count for 'Abnormal' Class: \n", supportVectorCount[0])
print("Support Vectors Count for 'Normal' Class: \n", supportVectorCount[1])

#Now take the Testing data:--------------------------------------------------------------------------
#Get the first 6 column names of the Test data and assign it to the Test Features (tFeatures)
tFeatures = list(test_data2.columns[:6])
#print("Features: \n", tFeatures)

#Get the data of the Test Features
testFeatures = test_data2[tFeatures]
#print("Test Features: \n", testFeatures.head())

#Use the classifier to predict the class of the Testing data
testPrediction = classifier.predict(testFeatures)
#print("Test Prediction: \n", testPrediction)

#For measuring parameters like Accuracy, Recall and Precision, compare this Predicted class with the Actul class of the Testing data
#Get the Actual class of the Testing data
testClass = list(test_data2["class"])
#print("Test Actual Class: \n", testClass)

#Creating a Confusion Matrix for the Actual Vs Prediction
confusionMatrix = confusion_matrix(testClass, testPrediction)
print("Confusion Matrix: \n", confusionMatrix)

#Measuring the Accuracy, Precision and Recall scores of the Classifier
#Accuracy is measured for the entire classifier, while Precision and Recall are computed for each class
print("Accuracy of the Classifier: %.2f" %(100*accuracy_score(testClass, testPrediction)),"%")

#Here we calculate the Precision and Recall for the positive class - "Normal"
#pos_label: The class to report if average='binary' and the data is binary. 
#If the data are multiclass or multilabel, this will be ignored; setting labels=[pos_label] and average != 'binary' will report scores for that label only.
#Here the class is binary. Hence, we provide pos_label = 'Normal', average="binary" as the parameters to calculate Precision and Recall
print("Precision value of the Positive Class ('Normal' Class): %.2f" %(100*precision_score(testClass, testPrediction, pos_label = 'Normal', average="binary")), "%")
print("Recall value of the Positive Class ('Normal' Class): %.2f" %(100*recall_score(testClass, testPrediction, pos_label = 'Normal', average="binary")), "%")

#Classification Report for each class:
target_names = ['Abnormal', 'Normal']
print("Precision and Recall for each class: \n",classification_report(testClass, testPrediction, target_names=target_names))

print("Support Vectors Count: \n", supportVectorCount)
print(supportVectorCount[0]+supportVectorCount[1])

#Visualizing the Confusion Matrix using the seaborn package
import seaborn as sns
fig, ax= plt.subplots(figsize=(8,6))

#When you set xticklabels=True, yticklabels=True in the Heatmap function, it automatically provides the labels for Confusion matrix with 0s, 1s and 2s in the Ascending order of the Class labels  
#Use this to find out the labels of the Confusion Matrix:
sns.heatmap(confusionMatrix, annot=True, linewidths=.5, xticklabels=True, yticklabels=True);

#After you find out the labels in the Confusion Matrix, remove the xticklabels & yticklabels parameters. Set the labels manually now
#annot=True to annotate cells - This provides labels(numbers) in the Confusion Matrix

#sns.heatmap(confusionMatrix, annot=True, linewidths=.5);
ax.xaxis.set_ticklabels(['Abnormal', 'Normal'], fontsize=12); ax.yaxis.set_ticklabels(['Normal', 'Abnormal'], fontsize=12);

#Setting the labels, title and tick marks
#Usually in the Confusion Matrix, the Y-Axis represents the True Values and X-Axis represents the Predicted Values
ax.set_title('Confusion Matrix', fontsize=20);
ax.set_xlabel('Predicted Labels', fontsize=14);
ax.set_ylabel('True Labels', fontsize=14); 

#ax.plot()
plt.show()

#print(type(supportVectors))
#Changing the Support Vectors array into a Data Frame
supportVectorsdf = pd.DataFrame(supportVectors)

#Storing the first column of the SupportVectors dataframe as col1df
col1df=  supportVectorsdf[supportVectorsdf.columns[0:1]]
#print("Support Vectors 1st Column: \n", col1df)
"""
#Minimum value of the column
#axis=0 for Columns & axis=1 for rows
col1min = col1df.min(axis=0)
print("Minimum value of Column 1: ",col1min)

#Maximum value of the column
col1max = col1df.max(axis=0)
print("Maximum value of Column 1: ",col1max)

#Range of the column
range1= col1max-col1min
print("Range of Column 1: ", range1)

#Summary statistics of the column
print("Column 1 Summary: \n", col1df.describe())"""

#Summary statistics of the column 1
print("Column 1 Summary: \n", col1df.describe())

#Range of Column 1
#axis=0 for Columns & axis=1 for rows
range1= col1df.max(axis=0)-col1df.min(axis=0)
print("Range of Column 1: ", range1)

########################################

col2df=  supportVectorsdf[supportVectorsdf.columns[1:2]]
#print("Support Vectors 2nd Column: \n", col2df)

#Summary statistics of the column 2
print("Column 2 Summary: \n", col2df.describe())

#Range of Column 2
range2= col2df.max(axis=0)-col2df.min(axis=0)
print("Range of Column 2: ", range2)

########################################

col3df=  supportVectorsdf[supportVectorsdf.columns[2:3]]
#print("Support Vectors 3rd Column: \n", col3df)

#Summary statistics of the column 3
print("Column 3 Summary: \n", col3df.describe())

#Range of Column 3
range3= col3df.max(axis=0)-col3df.min(axis=0)
print("Range of Column 3: ", range3)

########################################

col4df=  supportVectorsdf[supportVectorsdf.columns[3:4]]
#print("Support Vectors 4th Column: \n", col4df)

#Summary statistics of the column 4
print("Column 4 Summary: \n", col4df.describe())

#Range of Column 4
range4= col4df.max(axis=0)-col4df.min(axis=0)
print("Range of Column 4: ", range4)

########################################

col5df=  supportVectorsdf[supportVectorsdf.columns[4:5]]
#print("Support Vectors 5th Column: \n", col5df)

#Summary statistics of the column 5
print("Column 5 Summary: \n", col5df.describe())

#Range of Column 5
range5= col5df.max(axis=0)-col5df.min(axis=0)
print("Range of Column 5: ", range5)

########################################

col6df=  supportVectorsdf[supportVectorsdf.columns[5:6]]
#print("Support Vectors 6th Column: \n", col6df)

#Summary statistics of the column 6
print("Column 6 Summary: \n", col6df.describe())

#Range of Column 6
range6= col6df.max(axis=0)-col6df.min(axis=0)
print("Range of Column 6: ", range6)
                                        
