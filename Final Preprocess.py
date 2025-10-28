import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

#%% Load the dataset
file_path = 'Karnataka_data_with_texture_soil_dummies.xlsx'
df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

#%% Step 1: Exploratory Data Analysis (EDA)
print("\n" + "="*50)
print("STEP 1: EXPLORATORY DATA ANALYSIS")
print("="*50)

#%% Basic information
print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

#%% Check class distribution for District (if that's your target)
print("\nDistrict distribution:")
print(df['District'].value_counts())

#%% Correlation matrix for numerical features
numerical_cols = ['pH', 'OC (%)', 'N (kg/ha)', 'P (kg/ha)', 'K (kg/ha)']
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

#%% Distribution of numerical features
df[numerical_cols].hist(bins=15, figsize=(12, 8))
plt.suptitle('Distribution of Numerical Features')
plt.tight_layout()
plt.show()

#%% Step 2: Data Cleaning
print("\n" + "="*50)
print("STEP 2: DATA CLEANING")
print("="*50)

#%% Check for missing values
missing_values = df.isnull().sum()
print(f"Missing values:\n{missing_values[missing_values > 0]}")

#%% Handle missing values if any exist
if df.isnull().sum().sum() > 0:
    print("Handling missing values...")
    
    # Separate numerical and categorical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Impute numerical columns with median
    numerical_imputer = SimpleImputer(strategy='median')
    df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])
    
    # Impute categorical columns with mode
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    print("Missing values handled successfully!")

#%% Outlier detection and treatment using IQR method
print("\nOutlier detection using IQR method:")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers detected")
    
    # Cap outliers
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

#%% Step 3: Data Transformation
print("\n" + "="*50)
print("STEP 3: DATA TRANSFORMATION")
print("="*50)

#%% Identify categorical columns that need encoding
categorical_columns = ['District', 'Taluk/Block', 'Texture_Class', 'Soil_Type']
print("Categorical columns to encode:", categorical_columns)

#%% Create a copy of the dataframe for transformations
df_processed = df.copy()

#%% Label encoding for categorical variables
label_encoders = {}
for col in categorical_columns:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col} with {len(le.classes_)} unique values")

#%% Since we already have dummy variables, we'll use them directly
# Identify all feature columns (exclude identifier columns if any)
feature_columns = [col for col in df_processed.columns if col not in ['District', 'Taluk/Block', 'Texture_Class', 'Soil_Type']]

print(f"\nTotal features after encoding: {len(feature_columns)}")

#%% Simple one-line save of the processed dataframe
df_processed.to_excel('Karnataka_soil_processed_final.xlsx', index=False)

print("Processed dataset saved as 'Karnataka_soil_processed_final.xlsx'")

# Display basic info about the saved file
print(f"\nSaved dataset information:")
print(f"Rows: {df_processed.shape[0]}")
print(f"Columns: {df_processed.shape[1]}")
print(f"File size: {df_processed.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB (approx)")

# Show first few rows of saved data
print(f"\nFirst 3 rows of saved data:")
print(df_processed.head(3))

