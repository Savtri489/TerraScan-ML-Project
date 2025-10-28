import pandas as pd

#%% Step 1️⃣: Load your Excel file
df = pd.read_excel("Karnataka_data.xlsx")

#%% Step 2️⃣: Clean text columns (trim spaces, remove dots, fix inconsistent separators)
def clean_text_col(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r'\.$', '', regex=True)
        .str.replace(r'\s*/\s*', '/', regex=True)  # remove spaces around '/'
    )

for col in ['Texture_Class', 'Soil_Type']:
    if col in df.columns:
        df[col] = clean_text_col(df[col])

#%% Step 3️⃣: Split multi-category cells (e.g., "Red Sandy Loam/Red Clay Loam")
def split_and_expand(df, column, prefix):
    # Split each entry into multiple categories
    split_df = df[column].str.get_dummies(sep='/')
    # Prefix with column name (to avoid name clashes)
    split_df.columns = [f"{prefix}_{col.strip()}" for col in split_df.columns]
    return split_df

texture_dummies = split_and_expand(df, 'Texture_Class', 'Texture')
soil_dummies = split_and_expand(df, 'Soil_Type', 'Soil')

#%% Step 4️⃣: Merge the new dummy columns back to the original DataFrame
df_final = pd.concat([df, texture_dummies, soil_dummies], axis=1)

#%% Step 5️⃣: Print all the created dummy columns and their values
dummy_cols = texture_dummies.columns.tolist() + soil_dummies.columns.tolist()

print("🎯 All Created Dummy Columns and Their Values:\n")
for col in dummy_cols:
    print(f"{col}:")
    print(df_final[col].values)
    print("-" * 60)

#%% Step 6️⃣: Optional – preview first few rows
print("\n📊 Preview of Final DataFrame with Dummies:")
print(df_final.head())

#%% Step 7️⃣: Save the processed file safely
output_file = "Karnataka_data_with_texture_soil_dummies.xlsx"
df_final.to_excel(output_file, index=False)
print(f"\n✅ File saved as '{output_file}'")
