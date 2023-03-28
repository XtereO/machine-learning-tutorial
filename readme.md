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
