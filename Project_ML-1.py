
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
import time
from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile



df= pd.read_excel("D:/Dataset/Wisconsin_breast_cancer.xlsx")
df.head()


# In[9]:


df['diagnosis'] = np.where(df.diagnosis == 'M', 1, 0)


# In[17]:


df.columns


# In[21]:



y = df.diagnosis                          # M or B 
list = ['id','texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean',  'fractal_dimension_mean',
       'radius_se', 'texture_se',  'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se',  'symmetry_se',
       'fractal_dimension_se', 
       'perimeter_worst', 'area_worst', 
       'compactness_worst', 'concavity_worst',
       'symmetry_worst']
x= df.drop(list,axis = 1 )
x.head()

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[16]:


x.columns
# Create correlation matrix
corr_matrix =x.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
to_drop


# In[4]:


drop_list1 = ['perimeter_mean',
 'area_mean',
 'concavity_mean',
 'concave points_mean',
 'radius_se',
 'perimeter_se',
 'area_se',
 'compactness_se',
 'concavity_se',
 'concave points_se',
 'fractal_dimension_se',
 'radius_worst',
 'texture_worst',
 'perimeter_worst',
 'area_worst',
 'smoothness_worst',
 'compactness_worst',
 'concavity_worst',
 'concave points_worst',
 'fractal_dimension_worst']
x_1 = x.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 
x_1.head()


# In[19]:


f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[6]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[50]:



# Calcualte the Fisher Score (chi2) between each feature and target
fisher_score = chi2(x.fillna(0), y)
fisher_score


# In[49]:



p_values = pd.Series(fisher_score[1])
p_values.index = x.columns
p_values.sort_values(ascending=False)


# In[16]:


df.head()

y = df.diagnosis                          # M or B 
list = ['id', 'diagnosis']
x_3= df.drop(list,axis = 1 )
x_3.head()
x_3.describe()


# In[18]:


from collections import Counter
Counter(df["diagnosis"])


# In[20]:


vis1 = sns.pairplot(df, hue="diagnosis")
#fig = vis1.get_fig()
vis1.savefig("lda.png")


# In[22]:


np.set_printoptions(precision=5)

mean_vec = []
for i in df["diagnosis"].unique():
    mean_vec.append( np.array((df[df["diagnosis"]==i].mean()[:3]) ))
print(mean_vec)


# In[24]:



SW = np.zeros((3,3))
for i in range(1,2): #2 is number of classes
    per_class_sc_mat = np.zeros((3,3))
    for j in range(df[df["diagnosis"]==i].shape[0]):
        row, mv = df.loc[j][:3].reshape(3,1), mean_vec[i].reshape(3,1)
        per_class_sc_mat += (row-mv).dot((row-mv).T)
    SW += per_class_sc_mat


# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
style.use('fivethirtyeight')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  

# 0. Load in the data and split the descriptive and the target feature
# df = pd.read_csv('data/Wine.txt',sep=',',names=['target','Alcohol','Malic_acid','Ash','Akcakinity','Magnesium','Total_pheonols','Flavanoids','Nonflavanoids','Proanthocyanins','Color_intensity','Hue','OD280','Proline'])
df= pd.read_excel("D:/Dataset/Wisconsin_breast_cancer.xlsx")

df['diagnosis'] = np.where(df.diagnosis == 'M', 1, 0)
Y = df.diagnosis                          # M or B 
list = ['id','diagnosis']
X= df.drop(list,axis = 1 )
# X.head()

# X = df.iloc[:,1:].copy()
# target = df['target'].copy()
# X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.3,random_state=0) 
# 1. Instantiate the method and fit_transform the algotithm
# X = sc.fit_transform(X)
# LDA = LinearDiscriminantAnalysis(n_components=3) # The n_components key word gives us the projection to the n most discriminative directions in the dataset. We set this parameter to two to get a transformation in two dimensional space.  
# data_projected = LDA.fit_transform(X,Y)

# print(data_projected.shape)
# print(type(data_projected))
# print(data_projected)

lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(X, Y)
drA = lda.transform(X)

print(drA.shape)


# lda = LinearDiscriminantAnalysis()
# k_fold = cross_validation.KFold(len(A), 3, shuffle=True)
# print('LDA Results: ')
# for (trn, tst) in k_fold:
#     lda.fit(A[trn], y[trn])
#     outVal = lda.score(A[tst], y[tst])
#     #Compute classification error
# print('Score: ' + str(outVal))



# PLot the transformed data
# markers = ['s','x','o']
# colors = ['r','g','b']
# fig = plt.figure(figsize=(10,10))
# ax0 = fig.add_subplot(111)
# for l,m,c in zip(np.unique(Y),markers,colors):
#     ax0.scatter(data_projected[:,0][Y==l],data_projected[:,1][Y==l],c=c,marker=m)
    

