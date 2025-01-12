#this apps will be used to predict iris dataset
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#load iris dataset from sklearn
iris = load_iris()

X = iris.data
y = iris.target

#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#title
st.title("Iris Flower Prediction")

#input form numeric
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

#predict
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

#press button to predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(prediction)
    st.write(["setosa", "versicolor", "virginica"][prediction[0]])
