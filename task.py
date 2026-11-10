import pandas as pd
import numpy as np


np.random.seed(42)
n_rows = 150
data = {
    'Customer_ID': [f'CUST_{1000 + i}' if i % 12 != 0 else np.nan for i in range(n_rows)],
    'Join_Date': pd.date_range(start='2023-01-01', periods=n_rows, freq='2D'),
    'Age': [np.random.randint(18, 65) if i % 7 != 0 else np.nan for i in range(n_rows)],
    'Gender': np.random.choice(['Male', 'Female', 'Other'], size=n_rows, p=[0.48, 0.48, 0.04]),
    'City': np.random.choice(['Karachi', 'Lahore', 'Islamabad', 'Peshawar'], size=n_rows),
    'Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], size=n_rows),
    'Order_Count': np.random.randint(1, 30, size=n_rows),
    'Total_Spent': [round(np.random.uniform(500, 50000), 2) if i % 9 != 0 else np.nan for i in range(n_rows)],
    'Is_VIP': np.random.choice([True, False], size=n_rows, p=[0.2, 0.8])
}
df = pd.DataFrame(data)
print("Dataset Ready! Total rows:", len(df))

# Task 1 
print("\n--- Task 1 ---")
print(df.shape)
print(df.dtypes)
print(df.isna().sum())

# Task 2
df = df.dropna(subset=['Customer_ID'])
print("\n--- Task 2 ---")
print("Shape after drop:", df.shape)

# Task 3
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Total_Spent'] = df['Total_Spent'].fillna(df['Total_Spent'].mean())
print("\n--- Task 3 ---")
print(df.isna().sum())

# Task 4
df['Join_Date'] = pd.to_datetime(df['Join_Date'])
df['Join_Year'] = df['Join_Date'].dt.year
df['Join_Month'] = df['Join_Date'].dt.month_name()
print("\n--- Task 4 ---")
print(df[['Join_Date', 'Join_Year', 'Join_Month']].head())

# Task 5
df['Avg_Order_Value'] = df['Total_Spent'] / df['Order_Count']

# Task 6
df['High_Value_Customer'] = df['Avg_Order_Value'] > 2500
print("\n--- Task 5 & 6 ---")
print(df[['Total_Spent', 'Order_Count', 'Avg_Order_Value', 'High_Value_Customer']].head())
print(df['High_Value_Customer'].value_counts())

# Task 7
city_summary = df.groupby('City').agg(
    Total_Customers=('Customer_ID', 'count'),
    Avg_Total_Spent=('Total_Spent', 'mean'),
    Max_Order_Count=('Order_Count', 'max')
).reset_index()
print("\n--- Task 7 ---")
print(city_summary)

# Task 8
pivot = pd.pivot_table(df, index='Category', columns='Is_VIP',
                       values='Total_Spent', aggfunc='sum')
print("\n--- Task 8 ---")
print(pivot)

# Task 9
top5 = (df[df['Category'].isin(['Electronics', 'Clothing'])]
        .sort_values('Total_Spent', ascending=False)
        .head(5))
print("\n--- Task 9 ---")
print(top5[['Customer_ID', 'City', 'Category', 'Total_Spent']])

#  Task 10
vip = df[(df['Is_VIP']) & (df['City'].isin(['Karachi', 'Lahore']))]
print("\n--- Task 10 ---")
print(vip.groupby('City')['Order_Count'].agg(['count', 'mean']))

print("\n--- Extra for insights ---")
print(df.groupby('Is_VIP')[['Total_Spent','Order_Count','Avg_Order_Value']].mean())
print(df.groupby('Category')['Total_Spent'].sum())
print(df.groupby('City')['Total_Spent'].sum())
print("High value %:", df['High_Value_Customer'].mean()*100)
print(df.groupby('Is_VIP')['High_Value_Customer'].mean()*100)

# Task 11
# 1. Lahore has the most customers (41) and highest total revenue (~1.06M),
#    but Karachi has the highest avg spend per customer (~27,379).
# 2. VIP customers are NOT bigger spenders: avg Total_Spent is ~24,798 (VIP)
#    vs ~24,839 (non-VIP); only 17.9% of VIPs are High_Value vs 27.5% of non-VIPs.
# 3. Electronics (~997K) and Books (~1.00M) bring the most revenue;
#    Home is lowest (~614K). Overall 35 of 137 customers (25.5%) are High_Value.
