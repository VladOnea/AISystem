import pandas as pd

file_path = 'C:/Users/Vlad/Desktop/Licenta/Text labels.xlsx'
df = pd.read_excel(file_path)

df['ID'] = df['ID'].astype(str)
mirrored_df = df.copy()
mirrored_df['ID'] = 'mirrored_' + mirrored_df['ID']

combined_df = pd.concat([df, mirrored_df], ignore_index=True)

updated_file_path = 'C:/Users/Vlad/Desktop/Licenta/Updated_Text_labels.xlsx'
combined_df.to_excel(updated_file_path, index=False)

updated_file_path