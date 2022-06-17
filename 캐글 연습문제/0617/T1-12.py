import pandas as pd
df = pd.read_csv("../input/covid-vaccination-vs-death/covid-vaccination-vs-death_ratio.csv")
# print(df.head())

df2 = df.groupby(['country']).max()
df2 = df2.sort_values('ratio',ascending = False)
# print(df2.head())

df2 = df2[1:]
top = df2['ratio'].head(10).mean()
bottom = df2['ratio'].tail(10).mean()

ans = round((top - bottom),2)
print(ans)