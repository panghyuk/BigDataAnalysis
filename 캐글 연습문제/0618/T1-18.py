import pandas as pd

df = pd.read_csv("../input/bigdatacertificationkr/basic2.csv", parse_dates=['Date'])

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x: x >= 5)
# print(df.head())

weekday = df[(df['year'] == 2022) & (df['month'] == 5) & (~df['weekend'])]
weekend = df[(df['year'] == 2022) & (df['month'] == 5) & (df['weekend'])]

weekday_mean = weekday['Sales'].mean()
weekend_mean = weekend["Sales"].mean()
ans = weekday_mean - weekend_mean
print(round(ans,2))