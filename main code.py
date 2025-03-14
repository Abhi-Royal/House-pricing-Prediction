from pandas import *

data = read_csv("kc_house_data.csv")
print(data)
print(data.info())
print(data.describe())

# Data Pre-Processing / Data Cleaning

data['date'] = to_datetime(data['date'])
data['date'] = data['date'].astype(int)

data['yr_renovated'] = data['yr_renovated'].astype(bool)
print(data['yr_renovated'] )

data['waterfront'] = data['waterfront'].astype(bool)
print(data['waterfront'] )

data['view'] = data['view'].astype(bool)
print(data['view'] )

data.drop(columns=['id'], inplace = True)

print(data)

# Importing Linear Regression model

from sklearn import linear_model
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,mean_squared_log_error, mean_absolute_error, r2_score


X = data.drop(['price'], axis=1)
print(X)
y = data['price']
print(y)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

L_reg = linear_model.LinearRegression()
model = L_reg.fit(x_train,y_train)
Prediction = model.predict(x_test)
print(Prediction)

#print(accuracy_score(y_test,Prediction))

mse = mean_squared_error(y_test, Prediction)
mae = mean_absolute_error(y_test, Prediction)
r2 = r2_score(y_test, Prediction)
msle = mean_squared_log_error(y_test, Prediction)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
print("Mean Squared Logarithmic Error:", msle)

