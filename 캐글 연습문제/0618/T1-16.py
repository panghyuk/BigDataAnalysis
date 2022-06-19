import pandas as pd

# 데이터 불러오기
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")

new_df = df[df['f2'] == 0].sort_values('age').reset_index(drop = True)
# print(new_df)
new_df = new_df[:20]
std1 = new_df['f1'].var()
new_df['f1'] = new_df['f1'].fillna(new_df['f1'].min())
std2 = new_df['f1'].var()
ans = std1 - std2
print(round(ans,2))