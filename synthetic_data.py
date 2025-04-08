import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, TargetEncoder
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate a dataframe with 100 samples
n_samples = 500

# Generate data for each column
data = {
    # Categorical column with high cardinality (city names) - good for Target Encoding
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                              'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
                              'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte'], 
                             size=n_samples),
    
    # Categorical column with low cardinality - good for One Hot Encoding
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Home'], 
                                         size=n_samples),
    
    # Numeric column with outliers - good for Min Max Scaling
    'price': np.concatenate([
        np.random.uniform(10, 100, n_samples-5),  # Regular prices
        np.random.uniform(500, 1000, 5)           # Outliers
    ]),
    
    # Numeric column with normal distribution - good for Standard Scaling
    'customer_age': np.random.normal(35, 10, n_samples),
    
    # Date strings - need conversion to datetime64
    'purchase_date': [(datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d') 
                      for _ in range(n_samples)],
    
    # Continuous numeric column - can be used as is
    'quantity': np.random.randint(1, 10, n_samples)
}

# Create a target variable (e.g., purchase amount)
data['purchase_amount'] = (
    np.random.normal(0, 1, n_samples) +  # Base random component
    np.where(data['product_category'] == 'Electronics', 2, 0) +  # Electronics are more expensive
    np.where(data['product_category'] == 'Clothing', 1, 0) +     # Clothing is moderately priced
    data['price'] * 0.05 +                                       # Higher prices lead to higher purchase amounts
    data['quantity'] * 5 +                                       # More items lead to higher purchase amounts
    np.random.normal(0, 5, n_samples)                           # Random noise
)

# Create the DataFrame
df = pd.DataFrame(data)

df.to_csv('synthetic_data.csv', index=False)

print("Synthetic data generated and saved to 'synthetic_data.csv'.")
