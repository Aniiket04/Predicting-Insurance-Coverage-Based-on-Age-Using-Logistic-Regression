# Insurance Coverage Prediction Project

## Project Overview 

**Project Title : Insurance Coverage Prediction Project**
The goal of this machine learning project is to develop a predictive model that focuses on analyzing data related to insurance coverage, aiming to predict or explore trends in individuals purchasing insurance. 

## Objectives
1. **Analyzing Trends**: Investigating the relationship between individual's age and their likelihood of purchasing insurance.
2. **Predictive Modeling**: Using machine learning (logistic regression) to predict whether a person is likely to buy insurance based on their attributes, such as age.

## Project Structure

### 1. Importing Libraries
The notebook begins by importing essential Python libraries, including:
pandas for data manipulation
numpy for numerical operations
sklearn (from scikit-learn) for machine learning tools
matplotlib.pyplot or seaborn for data visualization
```python
import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
%matplotlib inline
```
%matplotlib inline is a jupyter notebook command which is used to display plots directly in the notebook output cells

### 2. Loading the Dataset
The given dataset is loaded using pandas.read_csv(). This dataset contains data about customer age and whether they have bought the insurance or not.
```python
df=pd.read_csv('Data2.csv')
df
```

### 3. Ploting the graph
Create a scatter plot to visualize the relationship between two variables: age and bought_insurance from the dataframe df
```python
plt.scatter(df.age,df.bought_insurance)
```

### 4. Train/Test Split
Divide the dataset into two parts:
a. Training set : Used to train the model
b. Testing set : Used to evaluate the model's performance on unseen data.
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.1)
x_test
```
test_size=0.1 means that 10% of the total data will be allocated for testing, and the remaining 90% will be used for training.

### 5.Model Training
Training a Logistics Regression model using sklearn.linear_model.LogisticsRegression
Fitting the model on the training data
```python
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
```

### 6. Model Predictions
Predictions on the held-out dataset.
```python
model.predict(x_test)
model.predict([[15]])
model.score(x_test,y_test)
```
The model.score() function evaluates the performance of a trained model.

## Conclusion
This project successfully demonstrated how logistic regression can be applied to predict the likelihood of buying insurance based on age. Age is a significant factor in predicting whether someone buys insurance.

## Author - Aniket Pal
This project is part of my portfolio, showcasing the machine learning skills essential for data science roles.

-**LinkedIn**: [ www.linkedin.com/in/aniket-pal-098690204 ]
-**Email**: [ aniketspal04@gmail.com ]


