import os
os.chdir(r"C:\Users\DELL\OneDrive\Desktop\Mini Project")

# %% 1. Imports and file path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


file_path = "Karnataka_data.xlsx"   # make sure this is in working directory

# %% 2. Load Excel file
df = pd.read_excel("Karnataka_data.xlsx")
print("✅ File loaded successfully.")
print("Initial shape:", df.shape)

# %% 3. Normalize column names
df.columns = (
    df.columns
    .str.strip()                    # remove extra spaces
    .str.replace(' ', '_')          # replace spaces with underscores
    .str.replace(r'[^\w\s]', '', regex=True)  # remove special chars
    .str.lower()                    # lowercase everything
)
print("✅ Column names standardized.\n")

# %% 4. Clean text-based columns
def clean_text_col(series):
    """Strip spaces, remove trailing dots, fix case, unify NA text."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r'\.$', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .replace({'nan': np.nan, 'NA': np.nan, 'N/A': np.nan, '-': np.nan})
        .str.title()  # Makes first letters uppercase (for names)
    )

# Apply cleaning for all object (text) columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = clean_text_col(df[col])

print("✅ Text columns cleaned.\n")

# %% 5. Specific known text replacements (adjust if needed)
if 'soil_type' in df.columns:
    df['soil_type'] = df['soil_type'].replace({
        'Red Soil/Black Soil': 'Red & Black Soil',
        'Red Sandy Loam/Black Soil': 'Red & Black Loam',
        'Lateritic Soil.': 'Lateritic Soil'
    })

if 'texture_class' in df.columns:
    df['texture_class'] = df['texture_class'].replace({
        'Clay Loam.': 'Clay Loam',
        'Sandy Loam.': 'Sandy Loam'
    })

print("✅ Specific replacements applied.\n")

# %% 6. Handle missing values
# Fill numerical NaNs with mean, categorical with mode
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("✅ Missing values handled.\n")

# %% 7. Remove duplicates
df.drop_duplicates(inplace=True)
print("✅ Duplicates removed.\n")

# %% 8. Detect and correct numeric format errors
# Convert object columns with numbers stored as text
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # skip non-numeric text

# %% 9. Handle outliers (optional)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[col] = np.clip(df[col], lower, upper)

print("✅ Outliers capped within 1.5×IQR.\n")

# %% 10. Encode categorical columns (if needed for ML)
encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])

print("✅ Categorical columns encoded.\n")

# %% 11. Standardize numeric columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("✅ Numeric columns standardized.\n")
# %%
# %% 12. Save cleaned data
df.to_excel("Cleaned_Karnataka_data.xlsx", index=False)
print("🎯 Cleaning complete. Cleaned file saved as 'Cleaned_Karnataka_data.xlsx'")
