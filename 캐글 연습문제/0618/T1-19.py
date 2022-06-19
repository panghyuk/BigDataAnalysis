import pandas as pd

df = pd.read_csv("../input/bigdatacertificationkr/basic2.csv", parse_dates=['Date'])

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

def event(x):
    if x['Events'] == 1:
        x['Sales2'] = x['Sales'] * 0.8
    else:
        x["Sales2"] = x["Sales"]
    
    return x

df = df.apply(lambda x: event(x), axis = 1)
# print(df.head())

df_22 = df[df['year'] == 2022]
sale_22 = df_22.groupby('month')['Sales2'].sum().max()
df_23 = df[df['year'] == 2023]
sale_23 = df_23.groupby('month')['Sales2'].sum().max()

ans = sale_22 - sale_23
print(int(round(abs(ans))))
