
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import scipy
from scipy import stats
import math
from itertools import product
import itertools
from itertools import combinations



df= pd.read_excel("C:/Users/mkahs/Desktop/Car_data.xlsx")

columns= df.columns
# df.columns
# df.at[2,'buying']

for var1 in df.columns:
    p = df[var1].unique()
    x_cols = [x for x in df.columns if x != var1]
#     print (var1)
    for var2 in x_cols:
        q = df[var2].unique()
#         print(var2)
        result=[]
        for i in range(len(p)):
            row=[]
        
            for j in range(len(q)):
                count=0
                
                for d in range(0, len(df)):
                    if((df.at[d,var1]==p[i]) & (df.at[d,var2]==q[j])):
                        count=count+1
#                         print(df.at[d,var1], "  &  ",df.at[d,var2], " COUNT : ", count )
                row.append(count)
#                 print("p[i]::",p[i]," count : ",count, " & ","q[j]:",q[j])
#                 print(df.at[d,var1], "  &  ",df.at[d,var2] )
#         print(row)    
            result.append(row)
#     pprint.pprint(result((len(p),len(q) )))
    np.array(result).reshape(len(p),len(q) )

    print(result)

