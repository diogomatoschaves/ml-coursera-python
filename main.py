

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from machinelearning_coursera.linear_regression import LinearRegression
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('ex1data2.txt', header=None)

df.columns = ['area', 'bedroom', 'price']

x = df[['area', 'bedroom']].to_numpy()
y = df.price.to_numpy()

# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# y = scaler.fit_transform(np.reshape(y, (-1, 1))).ravel()

reg = LinearRegression(max_iter=100000, learning_rate=0.001, normalize=True)
# reg = LinearRegression(strategy='normal_eq')
# reg = LinearRegression()
reg.fit(x, y)

if x.shape[1] == 1:
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(data=df, x='area', y='price')
    x_lin = np.linspace(5, 25, 100)
    y_lin = reg.predict(np.reshape(x_lin, (-1, 1)))

    plt.plot(x_lin, y_lin, color='r')

plt.show()