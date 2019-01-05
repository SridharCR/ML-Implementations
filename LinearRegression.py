import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn import datasets

data = datasets.load_boston()
#print(data.DESCR)
df = pd.DataFrame(data.data, columns = data.feature_names)
target = pd.DataFrame(data.target, columns = ["MEDV"])
#print(target)

X = df["RM"]
y = target["MEDV"]

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print(model.summary())
