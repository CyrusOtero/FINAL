import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

batting = pd.read_csv('Batting.csv')

print(batting.shape)


batting.plot.scatter("AB", "H")

x_train, x_test, y_train, y_test = train_test_split(batting.AB, batting.H, test_size = 0.2)
regr = LinearRegression()
regr.fit(np.array(x_train).reshape(-1,1), y_train)

preds = regr.predict(np.array(x_test).reshape(-1,1))

y_test.head()

preds