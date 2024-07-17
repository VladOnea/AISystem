import pandas as pd

def updateLabelsExcel(file_path, output_path):
    df = pd.read_excel(file_path)

    df['ID'] = df['ID'].astype(str)

    new_rows = []
    augmentations = [
        'mirrored',
        'translated',
        'rotated_90',
        'rotated_180',
        'rotated_270',
        'scaled_0.9',
        'scaled_1.1',
        'brightness_0.8',
        'brightness_1.2'
    ]

    for _, row in df.iterrows():
        original_id = row['ID']
        for aug in augmentations:
            new_row = row.copy()
            new_row['ID'] = f"{aug}_{original_id}"
            new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows)

    combined_df = pd.concat([df, new_rows_df], ignore_index=True)

    combined_df.to_excel(output_path, index=False)

    print(f"Updated Excel file saved to {output_path}")