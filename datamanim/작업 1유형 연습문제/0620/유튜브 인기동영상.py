import pandas as pd
pd.set_option("display.max_columns",None)
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/youtube.csv",index_col=0)
# print(df)
# print(df.info())

''' Q1 '''
ans = df.loc[df['channelId'].isin(df['channelId'].value_counts().head(10).index)]['channelTitle'].unique()
# ans = list(df.loc[df['channelId'].isin(df['channelId'].value_counts().head(10).index)]['channelTitle'].unique())
# print(ans)

''' Q2 '''
ans = df[df['dislikes'] > df['likes']]['channelTitle'].unique()
# print(ans)

''' Q3 '''
new_df = df[['channelTitle','channelId']].drop_duplicates()['channelId'].value_counts()
# print(new_df)
# print(len(new_df[new_df > 1]))

''' Q4 '''
df['trending_date2'] = pd.to_datetime(df['trending_date2'])
ans = df[df['trending_date2'].dt.day_name() == 'Sunday']['categoryId'].value_counts().index[0]
# print(ans)

''' Q5 '''
new_df = df.groupby([df['trending_date2'].dt.day_name(),'categoryId'],as_index = False).size()
ans = new_df.pivot(index = 'categoryId',columns = 'trending_date2')
# print(ans)

''' Q6 '''
new_df = df[df['view_count'] != 0].copy()
new_df['ratio'] = new_df['comment_count'] / new_df['view_count']
new_df = new_df.sort_values(by = 'ratio', ascending = False)
ans = new_df['title'].iloc[0]
# print(ans)

''' Q7 '''
ratio = (df['comment_count'] / df['view_count']).sort_values()
ans = df.iloc[ratio[ratio != 0].index[0]]['title']
# print(ans)

''' Q8 '''
new_df = df[(df['dislikes'] != 0) & (df['likes'] != 0)]
num = (new_df['dislikes']/new_df['likes']).sort_values().index[0]
ans = df.iloc[num]['title']
# print(ans)

''' Q9 '''
ans = df[df['channelId'] == df['channelId'].value_counts().index[0]]['channelTitle'].unique()[0]
# print(ans)

''' Q10 '''
ans = (df[['title','channelId']].value_counts() >= 20).sum()
print(ans)
