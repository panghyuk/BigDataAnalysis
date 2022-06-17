import pandas as pd
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
# print(df.head())

df_corr = df.corr()
# print(df_corr)
df_corr = df_corr[:-1]

corr_max = df_corr['quality'].max()
corr_min = df_corr['quality'].min()
# print(corr_max,corr_min)

ans = corr_max - corr_min
print(ans)