
# coding: utf-8

# In[1]:


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


# In[2]:


df['diagnosis'] = np.where(df.diagnosis == 'M', 1, 0)
df.columns


# In[14]:


y = df.diagnosis                          # M or B 
list = ['id', 'diagnosis']
       
x= df.drop(list,axis = 1 )
x.head()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[15]:


from sklearn.preprocessing import StandardScaler
conv = StandardScaler()
std_data = conv.fit_transform( x)


# In[10]:



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
lda = LinearDiscriminantAnalysis()
transformed_data = lda.fit_transform(x, y)

# transformed_data = pca.fit_transform( std_data )
print( transformed_data.shape )
print( lda.explained_variance_ratio_*100 )
print( lda.explained_variance_ )
print(transformed_data)


# x_test = lda.transform(x_test)

# plot the training data after transformation to see the distribution now
# benign_train_axis = []
# malignant_train_axis = []   
# for i in range(0,len(x_train)):
#     if(y_train[i]==0):
#         benign_train_axis.append(i)
#     else:
#         malignant_train_axis.append(i)  
# plt.plot(benign_train_axis, x_train[y_train==0], 'ko')
# plt.plot(malignant_train_axis, x_train[y_train==1], 'yo')
# plt.show()


# In[12]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # Classification
from sklearn.model_selection import cross_val_score # Cross validation for training data
from sklearn.metrics import confusion_matrix # Confusion matrix for test data


sc = StandardScaler()
x_train = sc.fit_transform(x_train.astype(np.float64))
if(x_test.shape[0] != 0):
    x_test = sc.transform(x_test.astype(np.float64))


# In[10]:


# Linear kernel SVM
classifier = LogisticRegression(random_state = 42)
classifier.fit(x_train, y_train)

# Applying k-Fold Cross Validation
print("Applying k fold cross validation ")
accuracies = cross_val_score(estimator=classifier,
                             X=x_train, y=y_train,
                             cv = 10, n_jobs=-1)
print("Cross validation accuracies :")
print(accuracies)
print("Cross validation mean :",accuracies.mean())
print("Cross validation std :",accuracies.std())


# In[16]:


target = df.diagnosis                          # M or B 
list = [ 'id', 'diagnosis']
       
raw_data= df.drop(list,axis = 1 )

from sklearn.preprocessing import StandardScaler
conv = StandardScaler()
std_data = conv.fit_transform( raw_data)


# In[17]:


# use PCA to reduce dimensionality
from sklearn.decomposition import PCA
pca = PCA(n_components=3,svd_solver='full')
transformed_data = pca.fit_transform( std_data )
print( transformed_data.shape )
print( pca.explained_variance_ratio_*100 )
print( pca.explained_variance_ )
print(transformed_data)

threshold = 0.65
for_test = 0
order = 0
for index,ratio in  enumerate (pca.explained_variance_ratio_):
    if threshold>for_test:
        for_test+= ratio
    else:
        order = index + 1
        break
print( 'the first %d features could represent 65 percents of the viarance' % order )
print( pca.explained_variance_ratio_[:order].sum() )
com_col = [ 'com'+str(i+1) for i in range(order) ]
com_col.append('others')
com_value = [ i for i in pca.explained_variance_ratio_[:order] ]
com_value.append( 1-pca.explained_variance_ratio_[:order].sum() )
com_colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightgrey', 'orange', 'white']
plt.figure( figsize=[4,4] )
plt.pie( x=com_value,labels=com_col,colors=com_colors,autopct='%.2f' )
plt.title( 'the first 3 components' )
plt.show()


# In[42]:


# to define the confusion_matrix and learning_curve
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix (  label,pred,classes = [0,1] ,cmap = plt.cm.Blues,title='confusion matrix' ):
    con_m = confusion_matrix( label,pred )
    plt.imshow( con_m,interpolation = 'nearest',cmap=cmap )
    plt.title(title)
    plt.colorbar()
    thres = con_m.max() / 2
    for j in range( con_m.shape[0] ):
        for i in range( con_m.shape[1] ):
            plt.text( i,j,con_m[j,i],
                      horizontalalignment = 'center',
                      color='white' if con_m[i,j]>thres else 'black')

    plt.ylabel( 'true label' )
    plt.xlabel( 'predicted label' )
    plt.xticks(  classes,classes )
    plt.yticks(  classes,classes )
    plt.tight_layout()
    
def print_matrix(  label,pred ):
    tn, fp, fn, tp = confusion_matrix( label,pred ).ravel()
    print( 'Accuracy rate = %.2f' %(( tp+tn )/( tn+fp+fn+tp )) )
    print('Precision rate = %.2f' % ((tp ) / (fp + tp)))
    print('Recall rate = %.2f' % ((tp ) / (fn + tp)))
    print('F1 score = %.2f' % ( 2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))) ))

def plot_ROC( label,pred ):
    from sklearn.metrics import roc_curve
    fpr, tpr,t = roc_curve( label,pred )
    plt.plot(fpr, tpr, label='ROC curve', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ')
    print( 'the threshold is ', t )
    plt.show()


from sklearn.model_selection import learning_curve
def plot_learning_curve( estimator,title,x,y,train_sizes = np.linspace(.1, 1.0, 5),n_job = 1 ):
    plt.figure( figsize=[4,4] )
    plt.title(title)
    plt.xlabel( 'Training examples' )
    plt.ylabel( 'Score' )
    train_size,train_score,test_score = learning_curve(estimator,x,y,n_jobs=n_job,train_sizes=train_sizes)
    train_scores_mean = np.mean(train_score, axis = 1)
    train_scores_std = np.std(train_score, axis = 1)
    test_scores_mean = np.mean(test_score, axis = 1)
    test_scores_std = np.std(test_score, axis = 1)
    plt.grid()
    plt.fill_between(train_size, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_size, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha = 0.1, color = "g")
    plt.plot(train_size, train_scores_mean, 'o-', color = "r",
             label = "Training score")
    plt.plot(train_size, test_scores_mean, 'o-', color = "g",
             label = "Cross-validation score")
    plt.legend(loc = "best")
    return plt


# In[43]:


# to pick the best estimator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(transformed_data, target, test_size = 0.12, random_state = random_seed)
logistic_reg = LogisticRegression( random_state=random_seed )
para_grid = {
            'penalty':['l1','l2'],
            'C':[0.001,0.01,0.1,1.0,10,100,1000]
            }
CV_log_reg = GridSearchCV( estimator=logistic_reg,param_grid=para_grid,n_jobs=-1 )
CV_log_reg.fit( X_train,y_train )
best_para = CV_log_reg.best_params_
print( 'the best parameters are ',best_para )


# In[44]:


# now using the best parameters to log the regression model
logistic_reg = LogisticRegression( C=best_para['C'],penalty=best_para['penalty'],random_state=random_seed )
logistic_reg.fit( X_train,y_train )
y_pred = logistic_reg.predict( X_test )

plot_confusion_matrix( y_test,y_pred )
plt.show( )
print_matrix(y_test,y_pred)
plot_ROC(y_test,y_pred)
plt.show( )


# In[31]:


y = df.diagnosis                          # M or B 
list = [  'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean',  'fractal_dimension_mean',
       'radius_se', 'texture_se',  'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'symmetry_se',
       'fractal_dimension_se',  
       'perimeter_worst', 'area_worst',
       'compactness_worst', 'concavity_worst', 
        'id', 'diagnosis']
       
x= df.drop(list,axis = 1 )
x.head()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[3]:


df['diagnosis'] = np.where(df.diagnosis == 'M', 1, 0)
y = df.diagnosis                          # M or B 
list = ['id','diagnosis']
x= df.drop(list,axis = 1 )
x.head()

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[4]:


x.columns
# Create correlation matrix
corr_matrix =x.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]
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


# In[5]:


f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


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
    

