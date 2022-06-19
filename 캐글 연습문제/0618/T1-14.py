import pandas as pd
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
# print(df.head())

new_df = df.groupby(['city','f4'])[['f5']].mean()
# print(new_df)
new_df.sort_values('f5',ascending = False,inplace = True)
# print(new_df.head(7))
ans = sum(new_df['f5'][:7])
print(round(ans,2))