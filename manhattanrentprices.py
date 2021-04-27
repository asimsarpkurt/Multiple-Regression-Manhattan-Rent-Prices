# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:33:33 2021
I have established a model and conducted multiple linear regression in order to predict the rent prices in Manhattan from the
data that gathered from Streeteasy with the actual rent prices in Manhattan. 
The main aim is to see that how independent variables impact prices which are specified as a list from the manhattan.csv file, named as a that contains the information about the properties of houses.
After performing multiple linear regression I also printed out the coefficients.
Coefficients are most helpful in determining which independent variable carries more weight. 
For example, a coefficient of -1.345 will impact the rent more than a coefficient of 0.238, with the former impacting prices negatively and latter positively.
I also checked the correlations between dependent and independent variables. 
Then I evaluated the accuracy of my multiple linear regression model with residual analysis.
@author: Sarp
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
streeteasy=pd.read_csv("C:/Users/DELL/.spyder-py3/manhattan.csv")
df=pd.DataFrame(streeteasy)
a=df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]
b=df[['rent']]
x_train,x_test,y_train,y_test=train_test_split(a,b,train_size=0.8,test_size=0.2,random_state=6)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
mlr=LinearRegression()
mlr.fit(x_train,y_train) #where our model learns from the data
y_predict=mlr.predict(x_test) 
example_apartment = [[1, 1, 620, 16, 1, 78, 0, 0, 1, 0, 0, 3, 1, 1]] #In our Manhattan model, we used 14 variables, so there are 14 coefficients
predict = mlr.predict(example_apartment) 
print("Predicted rent: $%.2f" % predict)
plt.scatter(y_test,y_predict)
plt.xlabel("Actual Rent Prices")
plt.ylabel("Predicted Rent Prices")
plt.title("Actual Rent vs Predicted Rent")
plt.show()
print("Coefficients:",mlr.coef_)
print("Intercept:",mlr.intercept_)
#Correlations between some of the dependent variables and indepent variable
plt.scatter(df[['size_sqft']],df[['rent']])
plt.show()
plt.scatter(df[['min_to_subway']],df[['rent']])
plt.show()
plt.scatter(df[['has_roofdeck']],df[['rent']])
plt.show()
plt.scatter(df[['building_age_yrs']],df[['rent']])
plt.show()
#Evaluating the model with the residual analysis
print('Train score:')
print(mlr.score(x_train,y_train))
print('Test score:')
print(mlr.score(x_test,y_test))
residuals=y_test-y_predict
plt.scatter(y_predict,residuals)
plt.title('Residual Analysis')
plt.show()
