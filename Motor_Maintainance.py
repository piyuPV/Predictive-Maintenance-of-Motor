#!/usr/bin/env python
# coding: utf-8

# In[250]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score


# In[251]:


df = pd.read_csv("feeds.csv")


# In[252]:


df.tail()


# In[253]:


#Removing Null values
update_df = df.drop([199, 200, 201,202])


# In[254]:


update_df


# In[255]:


update_df.describe()


# In[256]:


update_df.shape


# In[257]:


update_df.info()


# In[258]:


update_df.isnull().sum()


# In[259]:


x = update_df.iloc[:,[0,1,2,3,4]].values


# In[260]:


print(x)


# In[261]:


#using wcss

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++', random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[262]:


#plot elbow graph

sns.set()
plt.plot(range(1,11), wcss)
plt.title('Elbow point graph')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')
plt.show()


# In[263]:


#Optimum no. of cluster is 2

kmeans = KMeans(n_clusters = 2,init = 'k-means++', random_state = 0)

y = kmeans.fit_predict(x)


# In[264]:


print(y)


# In[265]:


#scater plot
#visualiszing ploting
plt.figure(figsize = (8,8))
plt.scatter(x[y == 0,0], x[y == 0,1], s = 50, c= 'yellow', label ='C1')
plt.scatter(x[y == 1,0], x[y == 1,1], s = 50, c= 'blue', label ='C2')


plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s= 100, c='green', label ='Centroid')
plt.show()


# In[266]:


sns.heatmap(update_df.corr(),annot=True)
plt.show()


# In[267]:


from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")


# In[268]:


classifier.fit(x,y)


# In[269]:


ypred= classifier.predict(x)
ypred


# In[270]:


accuracy_score(y, ypred)


# In[276]:


# input
#0 5.8	232	59.22	68	0
#1 5.1	229	65.23	68	0

#current = 5.8
#voltage = 232
#temperature = 59.22
#humidity = 68
#vibration = 0

current = 5.1
voltage = 229
temperature = 65.23
humidity = 68
vibration = 0


# In[277]:


test = [[current, voltage, temperature, humidity, vibration]]


# In[278]:


def give_pred(test):
    prediction = classifier.predict(test)

    if prediction == 0:
        return ('Your System Failed')
    else:
        return ('Your System Works')
    


# In[279]:


print(give_pred(test))


# In[ ]:




