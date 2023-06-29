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
    output.reset_index(drop=True, inplace=True)
    return output
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')
def model_by_feature(f1,f2):    
    input = train_data
    input = clean_data_by_column(input,f1)
    input = clean_data_by_column(input,f2)
    f1_vs_f2 = input[[f1,f2]]
    
    # Data Preparation
    X_train_selected = f1_vs_f2
    y_train = input['Survived']
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_selected, y_train, test_size=0.2, random_state=42)
    
    # Model Training and Evaluation
    logreg.fit(X_train, y_train)
    
    # Predicting on validation set
    y_pred = logreg.predict(X_val)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1_s = f1_score(y_val, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_s)
    
    plt.figure(figsize=(8,15))
    coefficients = logreg.coef_[0]
    intercept = logreg.intercept_

    # Define the x-axis range
    start = min(f1_vs_f2[f1])
    end = max(f1_vs_f2[f1])
    num_points = 100
    x = numpy.linspace(start, end, num_points)

    # Calculate the y-axis values using the logistic function
    y = -(intercept + coefficients[0] * x) / coefficients[1]
    
    # Generate x-axis and y-axis values
    x_values = numpy.linspace(min(f1_vs_f2[f1]), max(f1_vs_f2[f1]), 100)
    y_values = numpy.linspace(min(f1_vs_f2[f2]), max(f1_vs_f2[f2]), 100)
    
    # Plot the decision boundary
    X, Y = numpy.meshgrid(x_values, y_values)
    X_flattened = X.flatten().reshape(-1, 1)
    Y_flattened = Y.flatten().reshape(-1, 1)
    
    # Predict the class labels for the meshgrid points
    labels_pred = logreg.predict(numpy.hstack((X_flattened, Y_flattened)))

    # Reshape the predicted labels back to the meshgrid shape
    labels_pred = labels_pred.reshape(X.shape)
    plt.contourf(X, Y, labels_pred, levels=[-1, 0.5, 1], colors=['blue', 'red'], alpha=0.3)

    #plot the scatter plot
    plt.scatter(f1_vs_f2[f1], f1_vs_f2[f2], c=input['Survived'], cmap='coolwarm', label='Data Points', s=5)
    
    plt.xlim(min(f1_vs_f2[f1])-1, max(f1_vs_f2[f1])+1)
    plt.ylim(min(f1_vs_f2[f2])-1, max(f1_vs_f2[f2])+1)
    
    # Add labels, title, legend, etc. to the plot
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title(f'Logistic Regression Decision Boundary: {f1} vs {f2}')

    cbar = plt.colorbar()
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Died','Survived'])

    # Show the plot
    plt.show()
    
    # # clear the scatter plot
    plt.clf()

    test_input = clean_data_by_column(test_data,f1)
    test_input = clean_data_by_column(test_input,f2)
    test_f1_vs_f2 = test_input[[f1,f2]]
    
    X_test_selected = test_f1_vs_f2[[f1, f2]]
    plt.figure(figsize=(8, 6))

    # Create a meshgrid of points based on the range of 'Age' and 'Fare'
    x_min1, x_max1 = test_input[f1].min() - 1, test_input[f1].max() + 1
    y_min1, y_max1 = test_input[f2].min() - 1, test_input[f2].max() + 1
    xx1, yy1 = numpy.meshgrid(numpy.arange(x_min1, x_max1, 0.1), numpy.arange(y_min1, y_max1, 0.1))

    # Predict the class labels for the meshgrid points
    Z = logreg.predict(numpy.c_[xx1.ravel(), yy1.ravel()])

    # Reshape the predicted labels to match the meshgrid shape
    Z = Z.reshape(xx1.shape)

    # Plot the decision boundary
    plt.contourf(xx1, yy1, Z, alpha=0.5)

    # Scatter plot the test data points
    plt.scatter(test_input[f1], test_input[f2], s=5)

    # Add labels, title, legend, etc. to the plot
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title('Logistic Regression: Decision Boundary and Test Data')
    plt.colorbar(label='Prediction')

    # Show the plot
    plt.show()
    
model_by_feature("Age","Fare")
