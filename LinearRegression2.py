import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn import datasets, linear_model
data = datasets.load_boston()
#print(data.DESCR)
df = pd.DataFrame(data.data, columns = data.feature_names)
target = pd.DataFrame(data.target, columns = ["MEDV"])
#print(target)
X = df
y = target["MEDV"]

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

predictions = lm.predict(X)
print(predictions[0:5])
