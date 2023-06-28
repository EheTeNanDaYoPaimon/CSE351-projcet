#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as numpy
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[ ]:


test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')


# In[ ]:


display(train_data)


# In[ ]:


#age vs connection
#age in box plot
def non_outlier_bound(values):
    is_nan = numpy.isnan(values)
    values = numpy.array(values)[~is_nan]
    # Calculate the interquartile range (IQR)
    q1, q3 = numpy.percentile(values, [25, 75])
    iqr = q3 - q1

    # Define the lower and upper bounds for identifying outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound

ages = numpy.array(list(train_data["Age"]))
bounds = non_outlier_bound(ages)
input_survived = train_data.loc[bounds[0] <= train_data["Age"]].loc[train_data["Age"] <= bounds[1]].loc[train_data["Survived"] == 1]
input_died = train_data.loc[bounds[0] <= train_data["Age"]].loc[train_data["Age"] <= bounds[1]].loc[train_data["Survived"] == 0]
input_survived = input_survived[~numpy.isnan(input_survived["Age"])]
input_died = input_died[~numpy.isnan(input_died["Age"])]
ages_sp_survived = input_survived[["Age","SibSp","Parch"]]
ages_sp_survived = [ages_sp_survived["Age"],ages_sp_survived["SibSp"] + ages_sp_survived["Parch"]]
ages_survived = numpy.array(list(ages_sp_survived[0]))
sp_survived = numpy.array(list(ages_sp_survived[1]))
ages_sp_died = input_died[["Age","SibSp","Parch"]]
ages_sp_died = [ages_sp_died["Age"],ages_sp_died["SibSp"] + ages_sp_died["Parch"]]
ages_died = numpy.array(list(ages_sp_died[0]))
sp_died = numpy.array(list(ages_sp_died[1]))

plt.figure(figsize=(6, 15))


plt.scatter(sp_survived, ages_survived, color='red', marker='.')
plt.scatter(sp_died,ages_died,  color='blue', marker='.')


plt.ylabel('Age')
plt.xlabel('Total Number Of Relatives')
plt.title('plot')



# In[ ]:


display(train_data)


# In[ ]:


train_data['Age'].isnull().sum()


# In[ ]:


train_data_dropAgeNA = train_data
train_data_dropAgeNA.dropna(subset=['Age'], inplace=True)
train_data_dropAgeNA.reset_index(drop=True, inplace=True)
display(train_data_dropAgeNA)


# In[ ]:


train_data_dropAgeNA['Fare'].isnull().sum()


# In[ ]:


Fare_And_Age = train_data_dropAgeNA[['Age', 'Fare']]
display(Fare_And_Age)


# In[ ]:


# Data Preparation
# Assuming X_train_selected contains only 'Fare' and 'Age' columns
X_train_selected = Fare_And_Age
y_train = train_data['Survived']

# Splitting into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_selected, y_train, test_size=0.2, random_state=42)

# Model Training and Evaluation
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predicting on validation set
y_pred = logreg.predict(X_val)

# Evaluation metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[ ]:


coefficients = logreg.coef_[0]
intercept = logreg.intercept_

# Define the x-axis range for the graph
start = min(Fare_And_Age['Fare']) 
end = max(Fare_And_Age['Fare'])  
num_points = 100
x = numpy.linspace(start, end, num_points)

# Calculate the corresponding y-axis values using the logistic function
y = -(intercept + coefficients[0] * x) / coefficients[1]

plt.figure(figsize=(8,15))

# Plot the decision boundary
plt.plot(x, y, color='red', label='Decision Boundary')

# Plot the data points as a scatter plot
plt.scatter(Fare_And_Age['Fare'], Fare_And_Age['Age'], c=train_data['Survived'], cmap='coolwarm', label='Data Points', s=5)
plt.ylim(min(Fare_And_Age['Age']), max(Fare_And_Age['Age']))
# Add labels, title, legend, etc. to the plot
plt.xlabel('Fare')
plt.ylabel('Age')
plt.title('Logistic Regression Decision Boundary: Fare vs Age')
plt.legend()

# Show the plot
plt.show()


# In[ ]:


#def model_structure(column1, column2):
    

