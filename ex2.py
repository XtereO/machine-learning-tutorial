import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv("ex2.csv", encoding="Windows-1251")

model = PCA(svd_solver='full')
model.fit(data)

first_pc = model.components_[0]
second_pc = model.components_[1]

transformed_data = model.transform(data)
print(model.explained_variance_ratio_)
print(np.dot(transformed_data[0], first_pc))
print(np.dot(transformed_data[0], second_pc))
# for i, j in zip(transformed_data, data):
