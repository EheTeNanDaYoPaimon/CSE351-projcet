import pandas as pd
import numpy as numpy
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')
logreg = LogisticRegression()

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

def clean_data_by_column(data,feature):
    column = numpy.array(list(data[feature]))
    bounds = non_outlier_bound(column)
    output = data.loc[bounds[0] <= data[feature]].loc[data[feature] <= bounds[1]]
    output = output[~numpy.isnan(output[feature])]
    return output

def model_by_feature(f1,f2):

    input = clean_data_by_column(train_data,f1)
    input = clean_data_by_column(input,f2)
    useful_data = input[[f1,f2]]
    display(useful_data)
    print(useful_data[f2].max())
    # Data Preparation
    # Assuming X_train_selected contains only 'Fare' and 'Age' columns
    X_train_selected = useful_data
    y_train = input['Survived']

    # Splitting into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_selected, y_train, test_size=0.2, random_state=42)

    # Model Training and Evaluation
    logreg.fit(X_train, y_train)

    # Predicting on validation set
    y_pred = logreg.predict(X_val)

    # Evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1score = f1_score(y_val, y_pred)

    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1score)

    coefficients = logreg.coef_[0]
    intercept = logreg.intercept_

    # Define the x-axis range for the graph
    start = min(useful_data[f2]) 
    end = max(useful_data[f2])  
    num_points = 100
    x = numpy.linspace(start, end, num_points)

    # Calculate the corresponding y-axis values using the logistic function
    y = -(intercept + coefficients[0] * x) / coefficients[1]

    plt.figure(figsize=(8,15))

    # Plot the decision boundary
    plt.plot(x, y, color='red', label='Decision Boundary')

    # Plot the data points as a scatter plot
    plt.scatter(useful_data[f2], useful_data[f1], c=input['Survived'], cmap='coolwarm', label='Data Points', s=5)
    plt.ylim(min(useful_data[f1]), max(useful_data[f1]))
    # Add labels, title, legend, etc. to the plot
    plt.xlabel(f2)
    plt.ylabel(f1)
    plt.title('Logistic Regression Decision Boundary: Fare vs Age')
    plt.legend()

    # Show the plot
    plt.show()
model1 = model_by_feature("Age","Fare")
