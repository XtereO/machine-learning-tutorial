import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv("ex2_1.csv", encoding="Windows-1251")

model = PCA(svd_solver='full')
model = model.fit(data)
transformed_data = model.transform(data)

print(model.explained_variance_ratio_)
print(transformed_data[0])
