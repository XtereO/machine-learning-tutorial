# Method main components

## Why need
To reduce count of property (for example from 60 to 10). Sort by left properties

## How work
Take one line and then calculate angle by scalar product. Lines should have angle 90 between them and start from start coordinates. Lines must have length are 1. 

## Preparation
Find mean value each column, than minus from each one, so we get centralized values F. Then we have to find one vector x with length equal to 1 (Z = Fx). Find x and remember, that Z should be max and |x|=1. 


# Linear regression

## Motivation 
The Method main components is difficult in usage, so better option is linear regression. Based on multiple on one koeficent.

## How calculate main koeficents
y(x) = O1*x + O0
O1 = S((Yi-Ym)*(Xi-Xm))/S((Xi-Xm)^2)
O0 = Ym - O1*Xm

## How calculate SE
nf = n - (count freedom)
SE(y) = sqrt(S(Yi-y(Xi)^2)/nf) 
SE(O0) = SE(y)*sqrt((1/n) + (Xm^2/S((Xi-Xm)^2)))
SE(O1) = SE(y)*sqrt(1/S((Xi-Xm)^2))

## How calculate interval
ts - koeficent of Student (check the table)
O+ = O + ts*SE(O)
O- = O - ts*SE(O)

## How calculate Statistics mean
t = O1/SE(O1)
if ts>t then H0 accept, mean that equal show random values instead of real;else ts<t accept alternate theory Ha

## Usage in python
```
import pandas as pd
from sklearn.linear_regression import LinearRegression
from sklearn.model_selection import train_test_split

# preparing data
data = pd.read_csv("path.csv")
X = data.drop("Y", axis=1)
y = data["Y]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

# training 
model = LinearRegression()
model = model.fit(X_train, y_train)

# grade result
print(model.score(X_test, y_test))

# to predict new values
X_new_values = pd.read_csv("new_values.csv)
print(model.predict(X_new_values))

```