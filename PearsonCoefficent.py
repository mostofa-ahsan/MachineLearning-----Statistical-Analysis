
# coding: utf-8

# In[107]:


import numpy as np
import pandas as pd
from math import sqrt

df= pd.read_csv("C:/Users/mkahs/Desktop/winequality-white.csv",sep=';')
p=[]
q= []
for i in df.columns:
    s=0
    sigma_x=0
    sigma_y=0
    
    
    
    x = np.mean(df[i])
    y = np.mean(df['quality'])

    for j,k in zip(df[i],df['quality']):
        s = s + ((j-x)*(k-y))
        sigma_x= sigma_x+(j-x)**2
        sigma_y= sigma_y+(k-y)**2
        
    r = s/(sqrt(sigma_x*sigma_y))

    print("Pearson Coefficient between  [", i , "]  & Target Variable is: ", r )
    p.append(abs(r))
    q.append(i)
    #print(r)


# print(len(p))
# print(len(q))
print("#########################################################")
dict1= dict(zip(q,p))
sorted(dict1.values())
    
# p.sort()
for key, value in dict1.items():
    if (0.3>abs(value)>=0.1):
        print("The correlation between [",key,"] and Target variable is SMALL")
    if (0.5>abs(value)>=0.3):
        print("The correlation between [",key,"] and Target variable is MEDIUM")
    if (1>=abs(value)>=0.5):
        print("The correlation between [",key,"] and Target variable is LARGE")
    
        

