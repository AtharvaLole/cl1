import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data=pd.read_csv("datasets\\assignment_09\\part_a\\Car_Evaluation.csv")
le=LabelEncoder()
for col in data.columns:
    data[col]=le.fit_transform(data[col])
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
xtrain,xtest,ytrain,ytest=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model=RandomForestClassifier(random_state=42)
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
print(accuracy_score(ytest,pred))
print(confusion_matrix(ytest,pred))
