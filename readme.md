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

## How to calculate SE
nf = n - (count freedom)

SE(y) = sqrt(S(Yi-y(Xi)^2)/nf) 

SE(O0) = SE(y)*sqrt((1/n) + (Xm^2/S((Xi-Xm)^2)))

SE(O1) = SE(y)*sqrt(1/S((Xi-Xm)^2))

## How to calculate interval
ts - koeficent of Student (check the table)

O+ = O + ts*SE(O)

O- = O - ts*SE(O)

## How to calculate Statistics mean
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

# k-NN
Based on neighbourhouds look. Solve classifier problem.

## Metrics
Its answer for question how we calculate information.

### Evclid
d = sqrt((x1-x0)^2+(y1-y0)^2)

### Manhatan
d = |x1-x0|+|y1-y0|

### Chebyshaev
d = max|z1-z0|

## Algorithm
1 Read the data and choose the metric

2 Take new value and calculate distance by metric

3 Sort distance from new value

4 Pick k first data and check where is class more, that it

Also can calculate by distance and use weigth 1/d to sort more closer to new value correctly. And pick k odd number to prevent equal situation neigbourhouds.

## Usage in python
```
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# Get data and convert strings
data = pd.get_dummies(pd.read_csv("path.csv"))
X = data[["X1", "X2"]]
y = data["Y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Transform data to get more valuable values
scaler = MinMaxScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Model train and score
model = kNN(n_neighbors).fit(X_train, y_train)
f1_score(y_test, model.predict(X_test))
```

# Trust Bayevskiy Method
Based on Theory of Probability

## Example
ns = 11, no = 26, where ns - count spam letters, no - count ok letters. |V|=8, ws=69, wo=46, where V - count unique words, ws - words in spam, wo - words in ok. Take the word "win million dollars":

Fs = ln(ns/(ns+no))+ln((w1+1)/(ws+|V|+r))+ln((w2+1)/(ws+|V|+r))+ln((w3+1)/(ws+|V|+r))

Fo = ln(no/(ns+no))+ln((w1o+1)/(wo+|V|+r))+ln((w2o+1)/(wo+|V|+r))+ln((w3o+1)/(wo+|V|+r))

Where r - count words that missed in our list(let it be "dollars"), so r=1. wi - count word in spam, wio - count word in ok.

Ps = 1/(1+e^(Fo-Fs))

Po = 1/(1+e^(Fs-Fo))

Where Ps probability, that will in spam; Po - in ok.
