# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title("Dendogram")
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Fitting hierarchical clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

y_hc = hc.fit_predict(X)


plt.scatter(X[y_hc ==0, 0],X[y_hc ==0, 1], s = 100, c='red', label='Cluster1' )
plt.scatter(X[y_hc ==1, 0],X[y_hc ==1, 1], s = 100, c='blue', label='Cluster2' )
plt.scatter(X[y_hc ==2, 0],X[y_hc ==2, 1], s = 100, c='green', label='Cluster3' )
plt.scatter(X[y_hc ==3, 0],X[y_hc ==3, 1], s = 100, c='cyan', label='Cluster4' )
plt.scatter(X[y_hc ==4, 0],X[y_hc ==4, 1], s = 100, c='magenta', label='Cluster5' )
plt.title("Cluster of clients")
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()

