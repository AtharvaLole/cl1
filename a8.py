import pandas as pd
data=pd.read_csv("datasets\\assignment_08\\RealEstate_Prices.csv")
data.columns=data.columns.str.strip().str.replace(" ","_")
data=data.dropna()
data=data.drop_duplicates()

data=pd.get_dummies(data,drop_first=True)

data=data[data["X2_house_age"]<30]

print(data["X2_house_age"].mean())
print(data["X2_house_age"].max())
print(data["X2_house_age"].min())

q1=data["X2_house_age"].quantile(0.25)
q3=data["X2_house_age"].quantile(0.75)
iqr=q3-q1

data=data[
    (data["X2_house_age"]>=q1-1.5*iqr)&(data["X2_house_age"]<=q3+1.5*iqr)
]

data.to_csv("cleaned2.csv",index=False)
print("done")