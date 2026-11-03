import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv("datasets\\assignment_07\\part_a\\Iris.csv")
X=data.drop(columns=["Species","Id"],axis=1,errors="ignore")
wcss=[]

for i in range(1,11):
    model=KMeans(n_clusters=i,random_state=42,n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1,11),wcss,marker="o")
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.show()

model=KMeans(n_clusters=3,random_state=42,n_init=10)
data["Clusters"]=model.fit_predict(X)

plt.scatter(X.iloc[:,0],X.iloc[:,1],c=data["Clusters"])
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("k-means")
plt.show()

print(data.head())

