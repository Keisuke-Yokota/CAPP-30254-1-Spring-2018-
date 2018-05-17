import pandas as pd

filename1 = 'outcomes.csv'
filename2 = 'projects.csv'

o_df = pd.read_csv(filename1, header=0)
p_df = pd.read_csv(filename2, header=0)

df = p_df[(p_df.date_posted >= '2011-01-01')&(p_df.date_posted <= '2013-12-31')]
df.to_csv("project_2011_to_2013.csv")

u = df['state'].unique()
vc = o_df['fully_funded'].value_counts()

new_df = pd.merge(df, o_df, on='projectid', how='left')

new_df = df.drop(["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.1.1"],axis=1) 