import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#We apply the squared trick to linear regression.

df = pd.read_csv("perceptron_data.csv")

#The learning rate of the classifier
learning_rate = 0.01

#The number of iterations of gradient descent.
#TODO: Optimize number of rounds needed to train.
N = 10000

#Pick random line
#The initial range of random numbers the coefficients, a,b,c can take on.
#The range is from 0 to the mean value of the x and y data points.
#TODO: Improve the range of the initial random numbers
m = random.uniform(0,df.mean()[0])
b = random.uniform(0, df.mean()[1])
# ~ print("The initial random line is: "+str(round(a,3))+" * X1 + "+str(round(b,3))+" * X2 + "+str(round(c,3)))	

#Pick random point
rand = random.randint(0,df.shape[0]-1)
points = df.iloc[rand,:]
x = points[0]
y = points[1]

def gradient_descent(x,y,m,b):
	y_hat = m * x + b
	value = y - y_hat
	m = m + value * x * learning_rate
	b = b + value * learning_rate
	return (m,b)
	
for i in range(N):
	rand = random.randint(0,df.shape[0]-1)
	points = df.iloc[rand,:]
	x = points[0]
	y = points[1]

	
	updated_line = gradient_descent(x,y,m,b)
	m = updated_line[0]
	b = updated_line[1]

print(m)
print(b)
x = df.iloc[:,0]
y = df.iloc[:,1]
plt.scatter(x,y)

plt.plot(x, m*x + b,color='black')
plt.show()



