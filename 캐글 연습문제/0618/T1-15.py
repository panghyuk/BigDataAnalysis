import pandas as pd

# 데이터 불러오기
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
# print(df.head())

new_df = df.sort_values('age',ascending = False)[:20]
# print(new_df)
new_df['f1'] = new_df['f1'].fillna(new_df['f1'].median())
# print(new_df)
final_df = new_df[(new_df['f4'] == 'ISFJ') & (new_df['f5'] >= 20)]
# print(final_df)
ans = final_df['f1'].mean()
print(ans)