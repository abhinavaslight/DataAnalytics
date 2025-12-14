import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Dataset Loading
pd.set_option('display.max_columns', None)
df=pd.read_csv("Mall_Customers.csv")

print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

print("\nShape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns)

print("\nDataset info:")
print(df.info())

# Step 2: Data Cleaning
print("\nMissing values:")
print(df.isnull().sum())

print(df.drop_duplicates(inplace=True))

# Step 3: EDA
print("\nSummary statistics:")
print(df.describe())

cat_cols = df.select_dtypes(include='object').columns
if len(cat_cols) > 0:
    print("\nValue counts:")
    print(df[cat_cols[0]].value_counts())

corr = df.corr(numeric_only=True)
print("\nCorrelation matrix:")
print(corr)

# Step 4: Visualizations
sns.set_style("whitegrid")

# Scatter plot
plt.scatter(df['Annual_Income_(k$)'], df['Spending_Score'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('Annual Income vs Spending Score')
plt.show()

# Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Bar Chart: Gender-wise customer count
df['Genre'].value_counts().plot(kind='bar')
plt.title('Gender-wise Customer Distribution')
plt.xlabel('Gender')
plt.ylabel('Number of Customers')
plt.show()

# Box Plot: Spending Score by Gender
sns.boxplot(x='Genre', y='Spending_Score', data=df)
plt.title('Spending Score Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Spending Score')
plt.show()


