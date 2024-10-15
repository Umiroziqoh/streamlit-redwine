import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('winequality-red.csv')

dswinequality = load_data()

# Function to train the model
def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    regresi = LinearRegression()
    regresi.fit(x_train, y_train)
    return regresi, x_test, y_test

# Sidebar for user inputs
st.sidebar.title("Red Wine Quality Prediction")
st.sidebar.write("Input wine characteristics to predict the quality.")

# User input for the wine features
fixed_acidity = st.sidebar.slider('Fixed Acidity', float(dswinequality['fixed acidity'].min()), float(dswinequality['fixed acidity'].max()), 7.4)
volatile_acidity = st.sidebar.slider('Volatile Acidity', float(dswinequality['volatile acidity'].min()), float(dswinequality['volatile acidity'].max()), 0.7)
citric_acid = st.sidebar.slider('Citric Acid', float(dswinequality['citric acid'].min()), float(dswinequality['citric acid'].max()), 0.0)
residual_sugar = st.sidebar.slider('Residual Sugar', float(dswinequality['residual sugar'].min()), float(dswinequality['residual sugar'].max()), 2.0)
chlorides = st.sidebar.slider('Chlorides', float(dswinequality['chlorides'].min()), float(dswinequality['chlorides'].max()), 0.08)
free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', float(dswinequality['free sulfur dioxide'].min()), float(dswinequality['free sulfur dioxide'].max()), 15.0)
total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', float(dswinequality['total sulfur dioxide'].min()), float(dswinequality['total sulfur dioxide'].max()), 46.0)
density = st.sidebar.slider('Density', float(dswinequality['density'].min()), float(dswinequality['density'].max()), 0.9978)
pH = st.sidebar.slider('pH', float(dswinequality['pH'].min()), float(dswinequality['pH'].max()), 3.51)
sulphates = st.sidebar.slider('Sulphates', float(dswinequality['sulphates'].min()), float(dswinequality['sulphates'].max()), 0.56)
alcohol = st.sidebar.slider('Alcohol', float(dswinequality['alcohol'].min()), float(dswinequality['alcohol'].max()), 9.4)

# Main title
st.title("Red Wine Quality Prediction App")

# Display the dataset
st.write("### Wine Quality Dataset Preview:")
st.dataframe(dswinequality.head())

# Prepare the dataset for training
x = dswinequality.iloc[:, :-1].values  # Independent variables (all except 'quality')
y = dswinequality.iloc[:, -1].values   # Dependent variable ('quality')

# Train the model
regresi, x_test, y_test = train_model(x, y)

# Prediction
input_data = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free sulfur dioxide': [free_sulfur_dioxide],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

predicted_quality = regresi.predict(input_data)
st.write(f"### Predicted Wine Quality: {predicted_quality[0]:.2f}")

# Display regression plots for each feature
st.write("### Regression Plots:")
independent_columns = dswinequality.columns[dswinequality.columns != 'quality']
for column in independent_columns:
    plt.figure(figsize=(6, 4))
    sns.regplot(x=dswinequality[column], y=dswinequality['quality'], scatter_kws={'alpha': 0.5}, line_kws={"color": "red"})
    plt.xlabel(column)
    plt.ylabel('Quality')
    plt.title(f'Regression: {column} vs Quality')
    st.pyplot(plt.gcf())  # Display the plot in the Streamlit app

# Model Evaluation
st.write("### Model Evaluation:")
y_pred = regresi.predict(x_test)
r2 = regresi.score(x_test, y_test)
st.write(f"R-squared value of the model: {r2:.2f}")
