import matplotlib.pyplot as plt
import pandas as pd

csv=pd.read_csv("datasets\\assignment_02\\sales_part1.csv")
excel=pd.read_excel("datasets\\assignment_02\\sales_part2.xlsx")
json=pd.read_json("datasets\\assignment_02\\sales_part3.json")

#display
print("CSV DATA:",csv.head())
print("EXCEL DATA:",excel.head())
print("JSON DATA:",json.head())

#merge
data=pd.concat([csv,excel,json],ignore_index=True)

print("missing values:",data.isnull().sum())

#clean
data=data.drop_duplicates()
data=data.dropna()

#transform
data["Total"]=data["Sales"]

#sales analysis
print("total sales:",data["Total"].sum())
print("average sales:",data["Total"].mean())

sales=data.groupby("Category")["Total"].sum()
print("sales by product:",sales)

sales.plot(kind="bar",color="skyblue")

plt.ylabel("total")
plt.xlabel("product")
plt.show()