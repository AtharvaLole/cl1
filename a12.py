import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_csv("datasets/Retail_Sales_Data.csv")

# 2. Explore dataset
print(data.head())
print(data.columns)

# 3. Clean data
data = data.dropna()

# 4. Total sales by region
sales = data.groupby("Region")["Sales"].sum()
print("Region-wise Sales:\n", sales)

# 5. Bar chart
sales.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.show()

# 6. Top performing region
print("Top Region:", sales.idxmax())
print("Highest Sales:", sales.max())

# 7. Sales by region and category
category = data.groupby(["Region", "Category"])["Sales"].sum()
print(category)

# 8. Stacked bar chart
category.unstack().plot(kind="bar", stacked=True)
plt.title("Region and Category-wise Sales")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.legend()
plt.show()