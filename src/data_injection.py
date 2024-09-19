import numpy as np
import pandas as pd
import os

# Load dataset
df = pd.read_csv(r"C:\Users\redhu\OneDrive\Desktop\final_df.csv")

# Define data path
data_path = os.path.join('data', 'raw')

# Make directories
os.makedirs(data_path, exist_ok=True)

# Save the dataframe to CSV
df.to_csv(os.path.join(data_path, 'df.csv'), index=False)

