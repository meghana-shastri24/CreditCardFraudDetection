import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = "data/creditcard.csv" 
data = pd.read_csv(data_path)

# Quick overview of the dataset
print("First few rows of the dataset:")
print(data.head())

print("\nDataset info:")
print(data.info())

print("\nMissing values per column:")
print(data.isnull().sum())

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Check class distribution
print("\nClass distribution:")
class_counts = data['Class'].value_counts()
print(class_counts)

# Visualize class distribution
plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix')
plt.show()

# Distribution of transaction amounts
plt.figure(figsize=(6, 4))
sns.histplot(data['Amount'], bins=50, kde=True, color='purple')
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Time feature analysis
plt.figure(figsize=(6, 4))
sns.histplot(data['Time'], bins=50, kde=True, color='green')
plt.title('Transaction Time Distribution')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

# Boxplot for Amount by Class
plt.figure(figsize=(6, 4))
sns.boxplot(x='Class', y='Amount', data=data, palette='Set2')
plt.title('Transaction Amount by Class')
plt.xlabel('Class')
plt.ylabel('Amount')
plt.show()
