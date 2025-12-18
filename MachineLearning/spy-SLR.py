import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\K anusree\Downloads\Salary_Data.csv')

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test, y_test, color = 'red')  # Real salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Regression line from training set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_slope*12+c_intercept
print(y_12)

bias_score = regressor.score(x_train,y_train)
print(bias_score)

#STATS INTEGRATION TO ML
#MEAN
dataset.mean()

dataset['Salary'].mean()
#MEDIAN
dataset.median()
dataset['Salary'].median()
#MODE
dataset.mode()
#VARIANCE
dataset.var()
dataset['YearsExperience'].var()
#STD DEVIATION
dataset.std()
dataset['Salary'].std()

from scipy.stats import variation
variation(dataset.values)

variation(dataset['Salary'])
#CORRELATION
dataset.corr()
dataset["Salary"].corr(dataset['YearsExperience'])

#SKEWNESS

dataset.skew()
dataset["Salary"].skew()

#STANDARD ERROR

dataset.sem()

#Z-SCORE
import scipy.stats as stats
dataset.apply(stats.zscore)
stats.zscore(dataset["Salary"])

#SUM OF SQUARES ERROR(SSR)
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total =np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

#R2
r_square=1-SSR/SST
print(r_square)

bias = regressor.score(x_train,y_train)
print(bias)
variance = regressor.score(x_test,y_test)
print(variance)

import pickle
filename = 'linear_regression_model.pkl'
with open(filename, 'wb')as file:
    pickle.dump(regressor,file)
print("Model has been pickled and save as linear_regression_model.pkl")


