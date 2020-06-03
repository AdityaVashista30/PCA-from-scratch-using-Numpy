
# ### Load the Data and Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12,8)

# data URL: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
iris=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None)
iris.head()

iris.columns=['sepal_length','sapal_width','petal_length','petal_width','species']
iris.dropna(how='all',inplace=True)
iris.head()
iris.info()

# ### Visualize the Data
sns.pairplot(iris)

# ### Standardize the Data
x=iris.iloc[:,0:4].values
y=iris.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit_transform(x)

# ### Compute the Eigenvectors and Eigenvalues

covariance_matrix=np.cov(x.T)
print(covariance_matrix)

eigenValue,eigenVector=np.linalg.eig(covariance_matrix)
print("Eigen Values: \n",eigenValue)
print("Eigen Vectors: \n",eigenVector)

# ### Singular Value Decomposition (SVD)

eigenVecSVD,s,v=np.linalg.svd(x.T)
eigenVecSVD

# ### Picking Principal Components Using the Explained Variance
for val in eigenValue:
    print(val)

variance_explained=[(i/sum(eigenValue))*100 for i in eigenValue]
print(variance_explained)

cummulative_variance_explained=np.cumsum(variance_explained)
print(cummulative_variance_explained)

sns.lineplot(x=[1,2,3,4],y=cummulative_variance_explained)
plt.xlabel("Number of components")
plt.ylabel("Cumulative variance")
plt.title("Cumulative variance of all components of Iris")
plt.show()


# ###  Project Data Onto Lower-Dimensional Linear Subspace

projectionMatrix=(eigenVector.T[:2][:]).T
print("Projection Matrix: \n",projectionMatrix)

Xpca=x.dot(projectionMatrix)
for species in ('Iris-setosa','Iris-vesicolor','Iris-virginica'):
    sns.scatterplot(Xpca[y==species,0],
                   Xpca[y==species,1])
