import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,Ridge,LinearRegression

data=pd.read_csv("datasets\\assignment_03\\part_a\\uber.csv")

data=data.dropna()
data=data.drop(columns=["Unnamed: 0","key","pickup_datetime"],errors="ignore")

data=data[(data["fare_amount"]>0) & (data["fare_amount"]<100)]

print("correlation:",data.corr(numeric_only=True)["fare_amount"])

X=data.drop("fare_amount",axis=1)
Y=data["fare_amount"]

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,test_size=0.2,random_state=42
)

models={
    "linear regression":LinearRegression(),
    "ridge":Ridge(),
    "lasso":Lasso()
}

for name,model in models.items():
    model.fit(X_train,Y_train)
    print("name:",model)
    pred=model.predict(X_test)

    print("r2:",r2_score(Y_test,pred))
    print(np.sqrt(mean_squared_error(Y_test,pred)))
