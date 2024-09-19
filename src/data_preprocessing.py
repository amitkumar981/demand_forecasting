# Import libraries
import numpy as np
import pandas as pd
import os

# Load dataset (ensure the correct path to the raw data)
df = pd.read_csv('./data/raw/df.csv')

# Drop unnecessary columns
df.drop(columns=['Batch', 'Material Description', 'Order'], inplace=True)

# Convert 'Document Date' to datetime, handling mixed formats
df['Document Date'] = pd.to_datetime(df['Document Date'], dayfirst=True, errors='coerce')

# Extract the 'Year' from 'Document Date'
df['Year'] = df['Document Date'].dt.year

# Filter rows for the year 2023
df = df[df['Year'] == 2023]

# Convert 'Amount in LC' and 'Quantity' to float, removing commas and taking the absolute values
df['Amount in LC'] = df['Amount in LC'].replace(',', '', regex=True).astype(float).abs()
df['Quantity'] = df['Quantity'].replace(',', '', regex=True).astype(float).abs()

# Drop 'Document Date' and 'Year' columns
df.drop(columns=['Document Date', 'Year'], inplace=True)

# Create a pivot table summarizing 'Amount in LC' and 'Quantity' for each 'Material/Veh/Equip'
pivot_table = pd.pivot_table(df, 
                             values=['Amount in LC', 'Quantity'], 
                             index='Material/Veh/Equip', 
                             aggfunc='sum', 
                             fill_value=0)

# Reset index to turn the pivot table into a regular DataFrame
df = pivot_table.reset_index()

# Define data path for saving the processed data
data_path = os.path.join('data', 'processed')

# Make directories if they don't exist
os.makedirs(data_path, exist_ok=True)

# Save the processed DataFrame to CSV
df.to_csv(os.path.join(data_path, 'df_processed.csv'), index=False)