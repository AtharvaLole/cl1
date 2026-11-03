import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=pd.read_csv("datasets\\assignment_06\\Telecom_Customer_Churn.csv")
print(data.head())
print(data.info())
print(data.describe())

data["TotalCharges"]=pd.to_numeric(data["TotalCharges"],errors="coerce")
data=data.dropna()

data=data.drop_duplicates()

for col in data.select_dtypes(include="object").columns:
    data[col]=data[col].str.strip()

data["SeniorCitizen"]=data["SeniorCitizen"].astype(int)

q1=data["MonthlyCharges"].quantile(0.25)
q3=data["MonthlyCharges"].quantile(0.75)
iqr=q3-q1

data=data[
    (data["MonthlyCharges"]>=q1-1.5*iqr) & (data["MonthlyCharges"]<=q3+1.5*iqr)
]

data["avgmonthspend"]=data["TotalCharges"]/data["tenure"].replace(0,1)

data=data.drop("customerID",axis=1,errors="ignore")
data=pd.get_dummies(data,drop_first=True)

scaler=StandardScaler()
data[["tenure","MonthlyCharges"]]=scaler.fit_transform(data[["tenure","MonthlyCharges"]])

X=data.drop("Churn_Yes",axis=1)
y=data["Churn_Yes"]

xtrain,xtest,ytrain,ytest=train_test_split(
    X,y,test_size=0.2,random_state=42
)

print(xtrain.shape)
print(xtest.shape)

data.to_csv("cleaned_churn.csv",index=False)
print("data cleaning completed!")