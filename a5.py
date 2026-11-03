import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split

data=pd.read_csv("datasets\\assignment_05\\part_a\\Digits.csv")

X=data.drop("digit",axis=1)
y=data["digit"]

xtrain,xtest,ytrain,ytest=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model=SVC()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)

print("acc:",accuracy_score(ytest,pred))
print("cm:",confusion_matrix(ytest,pred))

