import numpy as np
import pandas as pd

data = pd.read_csv('rosstat.csv', delimiter=',', decimal=";", encoding="Windows-1251")

sorted_data = data.sort_values(by="salary", ascending=True)
print(sorted_data)
