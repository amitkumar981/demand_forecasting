import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Load the processed dataframe
df = pd.read_csv('./data/processed/df_processed.csv')

# Sort by Amount in LC in descending order
df.sort_values(by='Amount in LC', ascending=False, inplace=True)

# Calculate cumulative value and percentage
df['Cumulative Value'] = df['Amount in LC'].cumsum()
df['Cumulative Percentage'] = 100 * df['Cumulative Value'] / df['Amount in LC'].sum()

# Classification based on cumulative percentage
def classify(row):
    if row['Cumulative Percentage'] <= 70:
        return 'A'
    elif row['Cumulative Percentage'] <= 90:
        return 'B'
    else:
        return 'C'

df['Category'] = df.apply(classify, axis=1)

# Calculate unit price
df['unit price'] = df['Amount in LC'] / df['Quantity']

# Reorder columns for output
order = ['Material/Veh/Equip', 'Quantity', 'unit price', 'Amount in LC', 'Cumulative Value', 'Cumulative Percentage', 'Category']
df = df[order]

# Create a pivot table
pivot_table = pd.pivot_table(df, 
                             values=['Amount in LC', 'Quantity'], 
                             index='Category', 
                             aggfunc='sum', 
                             fill_value=0)

# Add numeric categories for plotting
df['Category_num'] = df['Category'].map({'A': 1, 'B': 2, 'C': 3})

# 3D plot setup
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Color mapping for categories
colors = {'A': 'r', 'B': 'g', 'C': 'b'}

# Plot 3D scatter plot
ax.scatter(df['Category_num'], df['Amount in LC'], df['Quantity'], 
           c=df['Category'].map(colors), s=100)

# Set labels and legend
ax.set_xlabel('Category (Numeric)')
ax.set_ylabel('Amount in LC')
ax.set_zlabel('Quantity')
ax.legend(['Category A', 'Category B', 'Category C'])

# Save 3D plot as PNG
plt.savefig('./data/processed/3d_category_scatter.png', format='png')

# Show the plot
plt.show()

# Convert 'Material/Veh/Equip' to object type
df['Material/Veh/Equip'] = df['Material/Veh/Equip'].astype(object)

# Select numerical features for clustering
X = df[['Amount in LC', 'Quantity', 'Cumulative Value', 'Cumulative Percentage']]

# Standardize the features for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Scatter plot for K-Means clustering
plt.figure(figsize=(10, 6))
plt.scatter(df['Amount in LC'], df['Quantity'], c=df['Cluster'], cmap='viridis', marker='o', s=100)

# Add title and labels
plt.title('Clusters based on Amount in LC and Quantity')
plt.xlabel('Amount in LC')
plt.ylabel('Quantity')

# Add color bar
plt.colorbar(label='Cluster')

# Save clustering plot as PNG
plt.savefig('./data/processed/kmeans_clusters.png', format='png')

# Show the plot
plt.show()
