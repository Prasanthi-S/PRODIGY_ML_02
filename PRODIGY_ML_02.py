#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[41]:


df=pd.read_csv("Mall_Customers.csv")


# In[42]:


df.head(10)


# In[43]:


df.shape


# In[44]:


df.info()


# In[45]:


X=df.iloc[:, [3,4]].values


# In[46]:


X


# In[47]:


from sklearn.cluster import KMeans
wcss=[]


# In[48]:


for i in range(1,11):
    kmeans = KMeans(n_clusters= i,init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[49]:


plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS Values')
plt.show()


# In[50]:


kmeansmodel = KMeans(n_clusters = 5,init='k-means++', random_state=0)


# In[51]:


y_kmeans = kmeansmodel.fit_predict(X)


# In[54]:


plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s=80, c="red", label='Customer 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s=80, c="blue", label='Customer 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s=80, c="yellow", label='Customer 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s=80, c="cyan", label='Customer 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s=80, c="black", label='Customer 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='magenta', label= 'Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Clusters of Customers')
plt.legend()
plt.show()


# In[ ]:




