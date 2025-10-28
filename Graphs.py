import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Load your dataset
import os
os.chdir(r"C:\Users\DELL\OneDrive\Desktop\Mini Project")
file_path = "Karnataka_data.xlsx"  # replace with your actual file path
df = pd.read_excel(file_path)

#%% List of columns with outliers
columns_with_outliers = ['OC (%)', 'N (kg/ha)', 'P (kg/ha)', 'K (kg/ha)']

#%% --- 1. BOX PLOTS ---
plt.figure(figsize=(12, 6))
for i, col in enumerate(columns_with_outliers, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

#%% --- 2. HISTOGRAMS ---
plt.figure(figsize=(12, 6))
for i, col in enumerate(columns_with_outliers, 1):
    plt.subplot(2, 2, i)
    plt.hist(df[col], bins=20, color='lightgreen', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#%% --- 3. SCATTER PLOTS (vs Index) ---
plt.figure(figsize=(12, 6))
for i, col in enumerate(columns_with_outliers, 1):
    plt.subplot(2, 2, i)
    plt.scatter(df.index, df[col], color='salmon', edgecolor='black')
    plt.title(f'Scatter Plot of {col}')
    plt.xlabel('Index')
    plt.ylabel(col)
plt.tight_layout()
plt.show()
