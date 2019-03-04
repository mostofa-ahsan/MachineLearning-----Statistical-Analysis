
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy
# from scipy.stats import chisqprob
# from scipy import chi2
from scipy import stats
import math
from math import sqrt
from itertools import product
import itertools
from itertools import combinations
from scipy.stats import chi2


df= pd.read_csv("C:/Users/mkahs/Desktop/test_2v.csv")
# df.reset_index(drop=True, inplace=True)


df.columns
# df.dropna(subset = ['bmi', 'smoking_status'])
# filter = df['smoking_status'] != ""
# df = df[filter]
# df.dropna(how='any')
df = df.replace('[]', np.nan)

df.dropna(how='any', inplace=True)
df.drop('id', axis=1, inplace=True)
df.columns = range(len(df.columns))
df = df.reset_index(drop=True)

for var1 in df.columns:
    p = df[var1].unique()
    print(type(var1))
#     print(df.at[2,10])
#     x_cols = [x for x in df.columns if x != var1]
#     print (var1)
#     for var2 in x_cols:
    q = df[9].unique()
#     print(q)
#         print(var2)
    result=[]
    for i in range(len(p)):
        row=[]
        
        for j in range(len(q)):
            count=0
                
            for d in range(0, len(df)):
#                 print(df.at[d,var1])
                
                if((df.at[d,var1]==p[i]) & (df.at[d,9]==q[j])):
                    count=count+1
#                         print(df.at[d,var1], "  &  ",df.at[d,var2], " COUNT : ", count )
            row.append(count)
#                 print("p[i]::",p[i]," count : ",count, " & ","q[j]:",q[j])
#                 print(df.at[d,var1], "  &  ",df.at[d,var2] )
#         print(row)    
        result.append(row)
#     pprint.pprint(result((len(p),len(q) )))
    np.array(result).reshape(len(p),len(q) )
    print(var1, " : ")

    print(result)
#     row_sums = [sum(row) for row in result]
#     col_sums = [sum(col) for col in result]
#     column_sums = [sum([row[k] for row in result]) for k in range(0,len(result[0]))]
#     nrows_sums = [sum([col[l] for col in result]) for l in range(0,len(result[1]))]
    c_s=np.sum(result, axis=0)
    r_s=np.sum(result, axis=1)
#     print(column_sums)
#     print(r_s)
#     print(c_s)
#     nrow= len(p)
#     ncol= len(q)
    chi_value=0
#     print(type(result))
    for x in range (len(p)):
        for y in range (len(q)):
            Oij=(result[x][y])
#             print("Oij : ",Oij)
#             print("c_s : ", c_s[y], " & r_s: ",r_s[x])
            Eij=((c_s[y]*r_s[x])/len(df))
#             print("Eij: ",Eij)
            z= (Oij-Eij)**2/Eij
#             print("C-V: ",z)
            chi_value=chi_value+z
            
    print("CHI-SQUARE VALUE between [",var1,"] & target is: ",chi_value)
#     print(type(len(p)),len(p)," & " ,len(q),type(len(q)))
    c=len(p)
    r=len(q)
    CI=.95
    
#     print(type(c),c," & " ,r,type(r))
    degree= (c-1)*(r-1)
    critical_value = chi2.ppf(CI, degree)
#     p_value = lambda chi_value, degree: stats.chi2.sf(chi_value, degree)
#     p_value= stats.chisqprob(critical_value,degree)
#     print("P-Value: ",p_value)
    if critical_value>chi_value:
        print(var1, " & target variable is Independent")
        
    m=min(c,r)
    cramer_v= sqrt(chi_value/(len(df)*(m-1)))
    print("Cramer V: ",cramer_v)
#     count_ER=0
#     for i in range (len(p)):
#         for j in range(len(q)):
#             print(result[i][j])
#             count_ER= count_ER+result[i][j]
# #         print(count_ER)
    

