import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

red=pd.read_csv("datasets\\assignment_01\\part_a\\winequality-red.csv",sep=";")
white=pd.read_csv("datasets\\assignment_01\\part_a\\winequality-white.csv",sep=";")

red["type"]=0
white["type"]=1

data=pd.concat([red,white])

X=data.drop(["type","quality"],axis=1)
Y=data["type"]

X=StandardScaler().fit_transform(X)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)

plt.scatter(X_pca[Y==0,0],X_pca[Y==0,1],label="red wine",color="red")
plt.scatter(X_pca[Y==1,0],X_pca[Y==1,1],label="white wine",color="blue")

plt.xlabel("pc1")
plt.ylabel("pc2")
plt.title("pca")
plt.legend()
plt.show()