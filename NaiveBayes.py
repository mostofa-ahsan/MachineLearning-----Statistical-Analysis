
import numpy
from sklearn.metrics import accuracy_score
from scipy.stats import norm

print 'Using Iris Buy Training data'
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
Xp=Xtrain[(C1>0),0:d]
Xn=Xtrain[(C1<0),0:d]

TPROBp=float(Xp.shape[0])/Xtrain.shape[0]
TPROBn=float(Xn.shape[0])/Xtrain.shape[0]
print 'Positive Probability is ',
print TPROBp
print ''
print 'Negative probability is ',
print TPROBn
print ''

#Calculate MEAN and SD
SDp = numpy.array(Xp).std(0)
MEANp = numpy.array(Xp).mean(0)
SDn = numpy.array(Xn).std(0)
MEANn = numpy.array(Xn).mean(0)

#Put MEAN and SD in numpy arrray
SDp=numpy.asarray(SDp)
MEANp=numpy.asarray(MEANp)
SDn=numpy.asarray(SDn)
MEANn=numpy.asarray(MEANn)

#Testing Starts

Xtest = numpy.loadtxt(testingFile)
nn=Xtest.shape[0]
Pos=[]
Neg=[]
P=[]
N=[]
Prediction=[]

for i in range (0,nn):
    for j in range(0,d):
        P.append(norm.pdf(Xtest[i,j], MEANp[j], SDp[j])) #Calculate the posiitve PDF
        N.append(norm.pdf(Xtest[i,j], MEANn[j], SDn[j]))  #Calculate the negative PDF
    PRODp=numpy.product(P) #Multiply the positive PDFs
    PRODn=numpy.product(N) ##Multiply the negative PDFs
    PROBp=((PRODp*TPROBp)/(PRODp*TPROBp+PRODn*TPROBn)) #Calculate posterior probabilty
    PROBn=((PRODn*TPROBn)/(PRODp*TPROBp+PRODn*TPROBn)) #Calculate posterior probabilty
    P=[]
    N=[]
    
    if (PROBp>PROBn):
        Prediction.append(1)
    else:
        Prediction.append(-1)

print 'Prediction is ' 
print Prediction
print ''
print 'The accuracy is :'
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
