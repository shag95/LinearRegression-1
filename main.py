import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
df=pd.read_csv("canada3.csv")
print(df)
per_capita_income=df.per_capita_income
year=df[['year']]
print(year)
print(per_capita_income)
reg=linear_model.LinearRegression()
reg.fit(year,per_capita_income)
plt.plot(year,per_capita_income)
plt.show()
print(reg)
print(reg.predict([[2020]]))
#print(p)
