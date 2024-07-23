import pandas as pd

df1 = pd.read_excel('submit.xlsx')
df2 = pd.read_excel('submit2.xlsx')

df1.rename(columns=lambda x: f"{x}_model1", inplace=True)
df2.rename(columns=lambda x: f"{x}_model2", inplace=True)

df1.rename(columns={'edition_id_model1': 'ID'}, inplace=True)
df2.rename(columns={'edition_id_model2': 'ID'}, inplace=True)
merged_df = pd.merge(df1, df2, on='ID')

merged_df['pheat'] = (merged_df['pheat_model1'] * 0.9 + merged_df['pheat_model2'] * 0.1) 

columns_to_drop = [col for col in merged_df.columns if col.endswith('_model1') or col.endswith('_model2')]
merged_df.drop(columns=columns_to_drop, inplace=True)
merged_df.rename(columns={'ID':'edition_id'},inplace=True)

merged_df.to_excel('results.xlsx', index=False)