# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the tumor.csv dataset using pandas.
2. Separate the dataset into features (X) and target variable (Y).
3. Split the dataset into training and testing sets using `train_test_split()`.
4. Train the Decision Tree Classifier model using the training data and predict the test data.
5. Calculate accuracy, generate the classification report, and display the confusion matrix.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: Suwasthika V
RegisterNumber: 212225040445
*/
#import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#load datasets
data=pd.read_csv('tumor.csv')
print(data.head())
print(data.columns)
#assigning
X=data.drop(columns=['Class'])
Y=data['Class']
#splitting datas
X_test,X_train,Y_test,Y_train=train_test_split(X,Y,test_size=0.3,random_state=42)
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
#calculate accuracy and print classification metrics
accuracy=accuracy_score(Y_test,Y_pred)
print("Name: Suwasthika V")
print("Register Number: 212225040445")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(Y_test,Y_pred))
conf_matrix=confusion_matrix(Y_test,Y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel=("Predicted")
plt.ylabel=("Actual")
plt.title("Confussion Matrix")
plt.show()
```

## Output:
<img width="858" height="372" alt="image" src="https://github.com/user-attachments/assets/cdbb6176-0948-457c-b7f3-3bea4006981b" />
<img width="987" height="108" alt="image" src="https://github.com/user-attachments/assets/e452f677-6460-4b2c-9b98-849a0bcc7e30" />
<img width="723" height="297" alt="image" src="https://github.com/user-attachments/assets/3c459c08-98b7-45e9-9836-1d29a4339431" />
<img width="713" height="555" alt="image" src="https://github.com/user-attachments/assets/d4302ba3-998d-4c66-954a-eac173200102" />

## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
