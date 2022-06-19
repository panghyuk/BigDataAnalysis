import pandas as pd

df = pd.read_csv("../input/bigdatacertificationkr/basic2.csv")
# print(df.info())
df['Date'] = pd.to_datetime(df['Date'])
# print(df.info())
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
ans = df[(df['year'] == 2022) & (df['month'] == 5)]['Sales'].median()
print(ans)