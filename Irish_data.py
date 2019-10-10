# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:51:34 2017

@author: Rahul
"""

import numpy
from sklearn.metrics import accuracy_score
from collections import Counter

print 'Using categorical program on Buy Training Data'
print ''
trainingFile = "BuyTraining.txt"
testingFile = "BuyTesting.txt"
Xtrain = numpy.loadtxt(trainingFile)
n = Xtrain.shape[0]
d = Xtrain.shape[1]-1
print 'Training data size is ',
print n, d
print ''

C1=Xtrain[:,d]
print 'Labels are',
print C1
print ''
Xp=Xtrain[(C1>0),0:d]
Xn=Xtrain[(C1<0),0:d]

a=Xp.shape[0]  #NUmber of Positive entries
b=Xn.shape[0] #Number of Negative entries

Ap=float(a)/n #Prob of Positive entries
print 'Positive Probability is ',
print Ap
print ''
Bp=float(b)/n #Prob of Negative entries
print 'Negative probability is ',
print Bp
print ''

Xtest = numpy.loadtxt(testingFile)
nn=Xtest.shape[0]
nd=Xtest.shape[1]-1

PP=[]
PN=[]
Prediction=[]

#Loop through each element in test data

for i in range (0,nn):  
    for j in range(0,d):
        value= Xtest[i,j]
        P =  Xp[:,j]
        P1 = Counter( P ) # Calculate the number of values and frequency in that column
        if 0 in P1.keys(): # Check if any value is zero, then increment by 1
            New=[x+1 for x in P1.values()]
        else:
            New=P1
        P2=New[value] #Extract the frequency of that value in the column
        PP.append(float(P2)/a) # Calculate the Prob given its a YES
        
        N =  Xn[:,j]
        P3 = Counter( N ) # Calculate the number of values and frequency in that column
        if 0 in P3.keys():# Check if any value is zero, then increment by 1
            Old=[x+1 for x in P3.values()]
        else:
            Old=P3
        P4=Old[value] #Extract the frequency of that value in the column
        PN.append(float(P4)/b) # Calculate the Prob given its a NO
    PP.append(Ap) # Add the YES prob to the list
    PN.append(Bp) # Add the NO prob to the list
    PRODp=numpy.product(PP) #Multiply all YES
    PRODn=numpy.product(PN) #Multiply all NO
    FPos=PRODp/(PRODp+PRODn) #Calculate YES Prob
    FNeg=PRODn/(PRODp+PRODn) #Calculate NO Prob
    
  
    if (FPos>FNeg):
        Prediction.append(1)
    else:
        Prediction.append(-1)
    
    PP=[]
    PN=[]

print 'Predicted values are: ', 
print Prediction
print ''
print 'Actual values are:    ',
print Xtest[:,-1]
print ''
print 'The accuracy is :',
print accuracy_score(Xtest[:,-1], Prediction)

tp = 0 #True Positive
fp = 0 #False Positive
tn = 0 #True Negative
fn = 0 #False Negative

for i in range(0,len(Prediction)):
    if (Prediction[i]==1 and Xtest[i,-1]==1):
        tp=tp+1
    elif (Prediction[i]==1 and Xtest[i,-1]==-1):
        fp=fp+1
    elif (Prediction[i]==-1 and Xtest[i,-1]==1):
        fn=fn+1
    elif (Prediction[i]==-1 and Xtest[i,-1]==-1):
        tn=tn+1


print ''
print 'True Positive is %s' %tp
print 'False Positive is %s'%fp 
print 'False Negative is %s'%fn 
print 'True Negative is %s'%tn

print '------------------------------------------'
print 'Precision is %s' %(float(tp)/(tp+fp))
print 'Recall is %s' %(float(tp)/(tp+fn))

print '------------------------------------------'
print 'DONE'
