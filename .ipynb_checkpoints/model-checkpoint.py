# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

print("Reading data from CSV file...")

df = pd.read_csv('clean_automobile.csv')

print("Reading data from CSV file COMPLETED!!!")

# MULTIPLE LINEAR REGRESSION

# A model using these variables as the 
# predictor variables.

Z = df[['make','body-style','drive-wheels','horsepower','city-mpg','highway-mpg','diesel','gas','aspiration-std','aspiration-turbo']]



regressor = LinearRegression()
print("Fitting regression model")
#Fitting model
regressor.fit(Z, df['price'])

print("Pickel dumpinp .pkl file...")
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

print(".pkl file created...")
# Loading model to compare the results
print("reading model from .pkl file...")
model = pickle.load(open('model.pkl','rb'))
print("predicting value...")
print(model.predict([['chevrolet','hatchback','front',94.5,38,43,0,1,1,0]]))