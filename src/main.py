#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets
import matplotlib.image as mpimg
import os
import scipy
import cv2
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, classification_report
import datetime
from skimage import io
np.random.seed(42)
import warnings
#warnings.filterwarnings('ignore')
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import sys



    


# In[ ]:


def train_test_split(X,Y,size=0.2):
        #Cross-validation -- to be done via k-fold later.
        from sklearn.model_selection import train_test_split  
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size)
        return X_train, Y_train,X_test, Y_test


# In[ ]:


# def scale(X,normalize=False,gaxis=1):
#     from sklearn.preprocessing import StandardScaler,MinMaxScaler
#     scaler = StandardScaler()
#     if(normalize):
#         X= sklearn.preprocessing.normalize(X,axis=gaxis)
#         return X
#     #print(X_S.shape)
#     X=scaler.fit_transform(X)
#     return X

def scale(X,testing=False,mode='standard',a=None,b=None):
    #X=np.nan_to_num
    X=np.array(X)
    if  mode=='scale':
        if(not testing):
            mx,mn=X.max(axis=0),X.min(axis=0)
        else:
            mx,mn=b,a
        mx=np.where(mx==mn,mx+1,mx)
        X=(X-mn)/(mx-mn)
        if(testing):return X
        return X,mn,mx
    elif mode=='standard':
        if(not testing):
            mean,std=X.mean(axis=0),X.std(axis=0)
        else:
            mean,std=a,b
        std=np.where(std==0,1,std)
        X=(X-mean)/std
        if(testing):return X
        return X,mean,std


# In[ ]:


def plotGraph(costs,fig_name,net=None,plot=True,Xtitle='Layer Count'):
    #plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    print('plott?',plot)
    aa=list(costs.values())   
    aa=np.array([list(i) for i in  aa])
    a1,a2,a3,a4,a5=aa.T #accuracy, cost
    plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
    #write after this line.
    plt.ylabel("Accuracy/Cost<Scaled-down by max={0}>".format(int(np.max(a2))))
    
    if type(net)==list:
        plt.title('DataSet={0}, model={1}, part={2}, task={3}'.format(net[0],net[1],net[2],net[3]))
        plt.xlabel(Xtitle)
    elif net is not None:
        print('yeah')
        plt.title('Dataset={1}, Layers={3}, Costs={2},\nActivators={0},batch_size={7}, ADAM={6}\nWeight-Init={4}, alpha={5},distribution={8}'.
                  format(net.activations,net.dataSetName,net.costName,net.layers,net.wInit,net.learningRate,net.doOp,net.batchSize,net.mode))
        plt.xlabel('no. of epochs')
    
    plt.subplot().plot(list(costs.keys()),a1,'*',label='Accuracy on Train Set')
    plt.subplot().plot(list(costs.keys()),a2/np.max(a2),'b', label='Cost of Train Data')
    plt.subplot().plot(list(costs.keys()),a3,'b--', label='Accuracy on Validation Set')
    plt.subplot().plot(list(costs.keys()),a4, label='f1-micro')
    plt.subplot().plot(list(costs.keys()),a5, label='f1-macro')
    
    plt.legend(loc='best', shadow=False)

    plt.savefig(fig_name)
    if not plot:
        pass
        #mpl.use('Agg')
    else:
        get_ipython().run_line_magic('matplotlib', 'inline')
    plt.show()
        


# In[ ]:


def oneHot(y,gClasses=None):
        S=list(set(y))
        if (gClasses):
            S=list(gClasses.values())
        classes={}
        #Y=np.zeros( (len(y),len(classes)))
        for i in range(len(S)):
            classes[i]=S[i]
        Y=[ [0 for i in range(len(S)) ] for _ in range(len(y))]
        for i in range(len(y)):
            #print(i,classes.index(y[i]))
            Y[i][S.index(y[i])]+=1
            #print(Y[i],classes.index(y[i]),i)
        if(gClasses):
            return Y
        return Y,classes


# In[ ]:


def preprocess(X,y,dataSetName,path=None,mode='standard',doScale=True,testing=False,classes=None):
        if(doScale):
            if(not testing):
                X,a,b=scale(X,testing,mode=mode)
                np.save('temp/A',a)
                np.save('temp/B',b)
                y,classes=oneHot(y)
                return X,y,classes
            else:
                a=np.load('{0}/A.npy'.format(path)).tolist()
                b=np.load('{0}/B.npy'.format(path)).tolist()
                X=scale(X,testing,'standard',a,b)
                y=oneHot(y,classes)
                return X,y
    
def BagOfWords(X,keys=None):
    #Converts word matrix to n X D matrix.
    #pre-process
    if keys is None:
        UniqueDict={}   
        for a in X:
            for t in a:
                    if t not in UniqueDict:
                            UniqueDict[t]=0
                    UniqueDict[t]+=1
        X_D=np.zeros((len(X),len(UniqueDict)),dtype='int32')
        keys=list(UniqueDict.keys())
        for a,c in zip(X, [ i for i in range(len(X))]):
            for t in a:
                    if t not in keys:
                            continue#security check
                    X_D[c][keys.index(t)]+=1

        return X_D,keys
    
    #else:
        X_D=np.zeros((len(X),len(keys)),dtype='int32')        
        for a,c in zip(X, [ i for i in range(len(X))]):
            for t in a:
                    if t not in keys:
                            continue#security check
                    X_D[c][keys.index(t)]+=1
        return X_D


# In[ ]:


class neuralNetwork:
    def __init__(self,X,y,classes=None,oneHot=True,dataSetName="",wInit=True,mode="gaussian",diminish=1,
                 hiddenlayers=[128,35],activations=['relu','tanh','soft-max'],cost='L2',
                 learningRate=[0.1,0.01,0.001],testModel=False):
        self.myactivators={'sigmoid':self.sigmoid,
                          'tanh':self.tanh,
                          'soft-max':self.softmax,
                            'relu':self.relu,
                            'swish':self.swish
                              }
        self.mycosts={'L2':self.L2_cost, 'cross_entropy':self.cross_entropy}
        if not testModel:
            self.dataSetName=dataSetName
            self.weightInit="random"
            self.X=X
            self.y=y
            self.classes=classes
            self.counter=0
            self.y=np.array(self.y)
            self.isOneHot=oneHot
            self.wInit=wInit
            self.mode=mode
            self.hiddenlayers=np.array(hiddenlayers)
            self.layers=list(hiddenlayers)
            self.layers.insert(0,self.X.shape[1])
            self.layers.append(self.y.shape[1])
            print(self.layers)
            self.activations=activations
            print(activations)
            self.methods=[ self.myactivators[i] for i in activations]
            self.learningRate=learningRate
            self.costName=cost
            self.cost=self.mycosts[cost]
            self.createLayers(diminish,self.wInit,self.mode)
            #self.initBias()
            self.initADAM()
            self.initADAMbias()
        

        
    def fitOnOtherDataSet(self,X,y,oneHot=True):
        self.X=self.scale(np.array(X))
        self.y=self.oneHot(y)
    '''        
    def dep_fit_train(X,y,self,batch_size=32,epochs=10):
        self.X=X
        self.y=y
        n=len(y)
        for epoch in range(epochs):
            print("epoch:{0}".format(epoch+1))
            inx=0
            while(inx<n):
                if(inx+batch_size>n):
                    Y=self.y[inx:]
                    X=self.X[inx:]
                else:    
                    Y=self.y[inx:inx+batch_size]
                    X=self.X[inx:inx+batch_size]
                
                self.train(X,Y)
                inx+=batch_size
            y_pred=self.getPredictions(X)
            print("Accuracy:",self.getAccuracy(self.y,y_pred))
    def dep_train(self,X,y,itr=1000):
        for _ in range(itr):
            if(_%100==0):
                print("training model at {0}th iteration".format(_))
            A,Z=self.feedForward(X,self.methods)
            self.backprop(A,Z,self.methods,y,self.cost,len(X))
    '''
    def testModel(self,X,y):
        yp=self.getPredictions(X)
        y=self.getOriginalClassIndex(np.array(y))
            
        print("Accuracy::",self.getAccuracy(yp,y))
        mi,mn=self.getF1Scores(y,yp)
        print("f1 Micro::",mi)
        print("f1 Macro::",mn)
        
        
    def getPredictions(self,X):
        z=X
        for i in range(len(self.layers)-1):
            w=self.weights[i]
            b=self.bias[i+1]#1xk
            a=np.add( np.dot(z,w) , b) #mxn nxk= mxk  -- wx+b
            z=self.methods[i](a)
        yp=np.argmax(z,axis=1)
        return yp
    def getOriginalClassIndex(self,z):#getOriginalClassIndex
        return np.argmax(z,axis=1)
    def getAccuracy(self,y_1,y_2):#Classification !
        return np.mean((y_1==y_2)) #CHECKPOINT, IF IT IS IN ONE HOT, THIS WILL LED WRONG RESULTS
    def ADAM_main(self,count,i,alpha,grad,bgrad):
        t=count+1
        #print('t:',t)
        self.weights[i],self.Am[i],self.As[i]=self.ADAM_updateWt(t,self.Am[i],self.As[i],self.weights[i],grad,alpha=alpha)
        #print("i:",i)
        if(i==-1):
                #print(self.adamM[i],self.adamM)
                self.bias[i],self.adamM[i],self.adamS[i]=self.ADAM_updateBias(t,self.adamM[i],self.adamS[i],self.bias[i],bgrad,alpha=alpha)
        else:
            self.bias[i+1],self.adamM[i+1],self.adamS[i+1]=self.ADAM_updateBias(t,self.adamM[i+1],self.adamS[i+1],self.bias[i+1],bgrad,alpha=alpha)
        
    def ADAM_WU(self,m,s,weight,grad,beta1=0.9,beta2=0.999,alpha=0.001,epsilon=1e-8):
        pass 
    
    def ADAM_updateWt(self,t,m,s,weight,grad,beta1=0.9,beta2=0.999,alpha=0.001,epsilon=1e-8):
        #print('aupwd:',t,np.max(s))
        m=beta1*m+(1-beta1)*grad
        s=beta2*s+(1-beta2)*np.multiply(grad,grad)
        mx=m/((1-beta1**t) )
        sx=s/(1-beta2**t)
        weight1=weight-alpha* np.divide(mx, (sx+epsilon)**(0.5) )
        #print(weight1==weight)
        return weight1,m,s
    
    def ADAM_updateBias(self,t,m,s,weight,grad,beta1=0.9,beta2=0.999,alpha=0.001,epsilon=1e-8):
        m=np.array(m)
        grad=np.array(grad)
        #print(beta1,m.shape,grad.shape)
        m=beta1*m+(1-beta1)*grad
        s=beta2*s+(1-beta2)*np.multiply(grad,grad)
        #print(beta1,t,beta1**t,m)
        mx=m/((1-beta1**t) )
        sx=s/(1-beta2**t )
        weight1=weight-alpha* np.divide(mx, (sx+epsilon)**(0.5) )
        #print(weight1==weight)
        return weight1,m,s
 
    def xav(self,L,K):
        return np.random.randn(L,K)*np.sqrt(1/L)
    def he(self,L,K):
        return np.random.randn(L,K)*np.sqrt(6/(L+K))
    
    def initWB(self,IP,OP,activator='relu',He=True,mode='gaussian'):
        print(IP,OP,activator)
        if He:
            _ = 1/(IP+OP)**0.5
            if activator in ('sigmoid','soft-max'):
                r, s = 6**0.5, 2**0.5
            elif activator=='tanh':
                r, s = 4*6**0.5, 4*2**0.5
            else: # relu or swish function
                r, s = 12**0.5, 2
            r, s = r*_, s*_
        else:
            r, s = 1, 1
        # Generating matrices
        if mode=='uniform':
            print('Mode -- Uniform')
            return 2*r*np.random.random((IP,OP))-r , 2*r*np.random.random((1,OP))-r
        elif mode=='gaussian':
            print('Mode -- gaussian')
            return np.random.randn(IP,OP)*s , np.random.randn(1,OP)*s
        else:
            print('Mode -- zeros')
            return np.zeros((IP,OP))

    
    def createLayers(self,diminish=1e0,He=True,mode='gaussian'):
        self.weights=[]
        self.bias=[[]]
        
        for i in range(len(self.layers)-1):
            #print(self.layers[i],self.layers[i+1],self.activations[i])
            w,b=self.initWB(self.layers[i],self.layers[i+1],self.activations[i],He=He,mode=mode)
            #self.weights.append( np.random.rand(self.layers[i],self.layers[i+1]) *diminish)
            #self.weights.append( np.zeros((self.layers[i],self.layers[i+1])))
            self.weights.append(w)
            self.bias.append(np.array(b))
        print('size of bias:',len(self.bias))
        
    def initADAM(self):
        self.Am=[]
        self.As=[]
        for i in range(len(self.layers)-1):
            
            #self.weights.append( np.random.rand(self.layers[i],self.layers[i+1]) *diminish)
            self.Am.append( np.zeros((self.layers[i],self.layers[i+1])))
            self.As.append( np.zeros((self.layers[i],self.layers[i+1])))
    def initADAMbias(self):
        self.adamM=[[]]
        for i in range(1,len(self.layers)):
            self.adamM.append( np.zeros((1,self.layers[i])))
        #print('size of bias:',len(self.adamM))
        self.adamS=[[]]
        for i in range(1,len(self.layers)):
            self.adamS.append( np.zeros((1,self.layers[i])))
        #print('size of bias:',len(self.adamS))
        
        
        
    def initBias(self):
        self.bias=[[]]
        for i in range(1,len(self.layers)):
            self.bias.append( np.zeros((1,self.layers[i])))
        print('size of bias:',len(self.bias))
    def getF1Scores(self,aa,bb):
        micron=sklearn.metrics.f1_score(aa,bb,average='micro')
        macron=sklearn.metrics.f1_score(aa,bb,average='macro')
        return micron, macron
            
    
    def train(self,initADAMS=True,doOp=False,batch_size=1,KKK=1,epochs=500,earlyStopping=False,X_val=None,y_val=None,printResults=False,minEpochs=100,patience=10): 
        #y_val== onehot vector.
        self.doOp=doOp
        acc_val,acc_main=0,1e100
        isUP=False #CHECKPOINT, m bola SGD CHALANE, TUM ADAM CHALAKE MAANOGE
        Costs={}
        n=self.X.shape[0]
        yp_ind=self.getOriginalClassIndex(self.y)
        self.batchSize=batch_size
        if(earlyStopping):
            X_val=np.array(X_val)
            #print(y_val)
            y_val=self.getOriginalClassIndex(np.array(y_val))
            #print(y_val)
            
        for _ in range(epochs):
            
            start_time=datetime.datetime.now()
            if(initADAMS):
                self.initADAM()
                self.initADAMbias()
            self.counter+=1
            if(printResults and (self.counter)%KKK==0):print((self.counter),end=' ')
            cost=0
            inx=0
            count=0
            while(inx<n):
                count+=1
                if(inx+batch_size>n):
                    Y_=self.y[inx:]
                    X_=self.X[inx:]
                else:    
                    Y_=self.y[inx:inx+batch_size]
                    X_=self.X[inx:inx+batch_size]
                
                A,Z=self.feedForward(np.array(X_),self.methods)
                cost+=self.backprop(A,Z,_,self.methods,np.array(Y_),self.cost,returnCost=True,doOp=doOp)
                inx+=batch_size
            
            if(earlyStopping):
                y_val_pred=self.getPredictions(X_val)
                #print(y_val_pred)
                tmp=self.getAccuracy(y_val,y_val_pred)
                if(isUP and tmp<acc_val and self.counter>minEpochs):
                    if(patience==10):
                        self.saveModel('{0}_patience_at_{1}'.format(self.dataSetName,self.counter))
                        #np.save('{0}_patience_0'.format(self.dataSetName))
                    patience-=1
                    if(patience==0):
                        break
                if(tmp>acc_val):
                    acc_val=tmp
                    isUP=True            
                    
            y_pred=self.getPredictions(self.X)
            acc_main=self.getAccuracy(yp_ind,y_pred)
            #acc_main=np.mean(y_pred==self.y_orig.T)
            mi,ma=self.getF1Scores(yp_ind,y_pred)
            
            end_time=datetime.datetime.now()
            if(printResults and(self.counter)%KKK==0):  
                
                
                print("Cost:",cost,"acc:",acc_main, 'validation_acc:',acc_val
                     ,'micro:',mi,'macro:',ma,'time:',end_time-start_time)
                
                pass
            if(earlyStopping):
                Costs[self.counter]=[acc_main,cost,acc_val,mi,ma]
            else:
                Costs[self.counter]=[acc_main,cost]
        return Costs
    
    def saveModel(self,modelName):
        np.save(modelName,[self.counter,self.weights,self.bias,self.activations,self.learningRate,self.layers,self.classes,self.costName])
    
    
    def loadModel(self,modelName):
        self.counter,self.weights,self.bias,self.activations,self.learningRate,self.layers,self.classes,self.costName=np.load(modelName).tolist()
        self.methods=[ self.myactivators[i] for i in self.activations]
        self.cost=self.mycosts[self.costName]
        
    def feedForward(self,X,method):
        '''
        Note X-- nxd -- represents n= images with d dim.
        W[0]=layer[0] X layer[1] or d x l1
        so a[1]= np.dot(X, W[0])
        z[1]=activator(a[1]) can be sigmoid/relu/tanh/squish etc...
        '''
        Z=[X]
        A=[[]]
        for i in range(len(self.layers)-1):
            w=self.weights[i]#nxk  #YAHAN TU BIAS & WEIGHT K ALAG INDEX LIYE HO, but BACK_PROP m SAME, CHAKKAR kya h
            b=self.bias[i+1]#1xk
            a=np.add( np.dot(Z[i],w) , b) #mxn nxk= mxk  -- wx+b
            A.append(np.array(a))
            z=method[i](a)
            Z.append(np.array(z))
            
            #print("A Z shape",A[-1].shape,Z[-1].shape)
        return A,Z
    
    def backprop(self,A,Z,count,method,y,cost,optimizer=ADAM_main,printCost=False,returnCost=True,doOp=True):
        #here it should be no. of samples--batch size
        #print("z,y,shapes",Z[-1].shape,y.shape)
        m=Z[0].shape[0]
        E=cost(Z[-1],y)
        if(printCost):
            print("COST:",E)
        dEdOout=cost(Z[-1],y,derivative=True)# CHECKPOINT, why so 1D-vector. Actually its m X 1d-vector
        dOoutdOin=method[-1](A[-1],derivative=True)#1D-vector
        dOindw=Z[-2]#HlastOut 1D-vector
        #print("dOindw nx14",dOindw.shape)
        #####
        dEdOin=dEdOout*dOoutdOin#This is right  
        #print('dEdOin shape',dEdOin.shape)
        '''
        n=1
        dEdw=np.matmul(dOindw.reshape(-1,n),dEdOin.reshape(n,-1)) # (Hlast,n)* (n,Oin) -- can cause problem for batch-grad
        '''
        dEdw=np.dot(dOindw.T,dEdOin) # (Hlast,n)* (n,Oin) -- can cause problem for batch-grad
        #print('dedw shape',dEdw.shape)
        if(doOp):
            dEdw=dEdw/np.where(np.mean(dEdw)==0,1,np.mean(dEdw))
        
            optimizer(self,count,-1,self.learningRate[-1],dEdw,np.mean(dEdOin,axis=0))#sum?
        else:
            self.weights[-1]-=self.learningRate[-1]*(dEdw/m)
        
            self.bias[-1]-=self.learningRate[-1]*np.mean(dEdOin,axis=0)
        #print('dedw:{0}\ndEdOin:{1}\ndEdOout:{2}\ndOoutdOin:{3}'.format(dEdw,dEdOin,dEdOout,dOoutdOin))
        #### Do general Recursion Now.
        #Call dEdOin as delta
        delta= dEdOin
        #print('delta:',delta.shape)
          
        # Weights=[in * h1, h1 *h2, h2 * hlast, hlast * out]
        # Already Calculated hlast* out or weights[-1]
        for i in range(len(self.weights)-2,-1,-1):
            '''
            size(Z)=size(A)=size(weights)+1
            '''
            dHoutdHin=method[i](A[i+1],derivative=True)
            dHindw=Z[i]
            #dHindw=np.tile(dHindw.reshape(-1,1),self.weights[i].shape[1])
            #print('dhindw',dHindw)
            #Need to find dEtotaldHout=dEtotal_dOin*dOin_dHout
            dEtotaldHout=np.dot(delta,self.weights[i+1].T)
            #print()
            dEdHin=np.multiply(dEtotaldHout,dHoutdHin)     #refraining use of Etotal. jUst E now. 
            #print("e/hout",dEtotaldHout,"\nhout/hin",dHoutdHin,"\ne/hin",dEdHin)
            dEdw=np.dot(dHindw.T,dEdHin) # (Hlast,1)* (1,Oin)
            #print(dEdw.shape,dEdw)
            if(doOp):
                dEdw=dEdw/np.where(np.mean(dEdw)==0,1,np.mean(dEdw))
                optimizer(self,count,i,self.learningRate[i],dEdw,np.mean(dEdHin,axis=0))
            else:
                self.weights[i]-=self.learningRate[i]*(dEdw/m)
            
                self.bias[i+1]-=self.learningRate[i]*np.mean(dEdHin,axis=0)
            delta=dEdHin
            #print('delta:',delta)
        return np.mean(E)
            
        '''
        np.repeat(z,3,axis=0).reshape(3,3)-x
        try to make it for mini-batch over stochastic
        '''
        
    def softmax(self,a,derivative=False):
        z=np.exp(a-a.max(axis=1,keepdims=True))
        if(derivative):
            su=np.sum(z,axis=1).reshape(-1,1)#try to use np.sum(s,axis=1 for row-wise sum ; 0 for col-wise sum)
            t=su-z
            tsq=np.sum(z,axis=1).reshape(-1,1)**2
            z=np.multiply(t,z)
            return z/np.maximum(tsq,1e-6)
        return z/np.sum(z,axis=1).reshape(-1,1)
    
    def relu(self,a,derivative=False):
        if(derivative==True):
            return (np.sign(a)>0)*1
        
        return np.maximum(a,0)
    def swish(self,a,derivative=False):
        z=a* self.sigmoid(a)
        if derivative:
            z=z+self.sigmoid(a)*(1-z)
        return z
    def sigmoid(self,a,derivative=False):
        
        #z= np.array(1/(1+ np.exp(np.multiply(a,-1))) )
        #try:
        # Prevent overflow.
        a = np.clip( a, -500, 500 )
        f = lambda x: 1/(1+np.exp(-x))
        g = lambda x: np.exp(x)/(1+np.exp(x))
        z= np.where(a>=0,f(a),g(a))
#         if(a.any()>=0):
#             z= np.array(1/(1+ np.exp(-a)) )
#         else:
#             z= np.array(1/(1+ np.exp(a)) )
# #         except:
# #             print('Sigmoid error:{0} at epoch:{1} layer:{2}'.format(np.max(-a),self.counter,a.shape))
# #             z=a
        if(derivative ==True):
            return np.multiply(z,(1-z))
        return z
    def tanh(self,a,derivative=False):
        #z=(2/(1+np.exp(-2*a))) -1
        if(derivative):
             return (1 - (np.tanh(a) ** 2)) 
        return np.tanh(a)
    
    def Identity(self,a,derivative=False):
        return a
    
    def L2_cost(self,A,B,derivative=False):
        A=np.array(A)#OUT
        B=np.array(B)#Actual output y
        #print('cost:',A.shape, B.shape)
        C=A-B
        if(derivative):
            return C
        return np.sum(C**2,axis=1)
    
    def cross_entropy(self,CalcOutput,trueOutput,derivative=False):
        
        A=np.array(CalcOutput)#OUT
        B=np.array(trueOutput)#Actual output y
        A=np.where(B!=1,A+np.e,A)# 0log0
        A=np.where(np.logical_and(B==1,A==0),A+1e-8,A)#1log0
        #print('cost:',A.shape, B.shape)
        if(derivative):
            return A-B
        return np.sum(-1*B*(np.log(A)),axis=1)
    


# In[ ]:


def loadDataSet(folder,rgb=False,itr=None):
    images = []
    count=0
    for filename in os.listdir(folder):
        count+=1
        #img=scipy.ndimage.imread(os.path.join(folder, filename), mode='L')
        img = (mpimg.imread(os.path.join(folder, filename)))
        if(rgb):
            img=rgb2gray(img)
            
        img=np.array(img)
        #img=img.reshape(-1,1)
        if img is not None:
            images.append(img)
        if itr is not None and count>itr:
            break
        
    return images

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
def loadDataSet2(folder,IMG_SIZE=100,as_gray=True,itr=None):
    images = []
    count=0
    for filename in os.listdir(folder):
        count+=1
        #img=scipy.ndimage.imread(os.path.join(folder, filename), mode='L')
        #img = io.imread(os.path.join(folder, filename),as_gray=as_gray)
        img = cv2.imread(os.path.join(folder, filename),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))    
        img=np.array(img)
        #img=img.reshape(-1,1)
        if img is not None:
            images.append(img)
        if itr is not None and count>itr:
            break
        
    return images

def LoadDataForCSV(fileName):
    f=open(fileName,'r')
    X=np.array([[float(i) for i in line.split(' ')] for line in f])
    f.close()
    return X



def LoadDataForTXT(fileName):
    f=open(fileName,'r')
    X=np.array([[str(i) for i in line.strip().split(' ')] for line in f])
    f.close()
    return X


# In[ ]:



#X_train,y_train,X_val,y_val,classes=MNIST()


# In[ ]:





# In[ ]:


#X_train.shape,y_train.shape, X_val.shape, y_val.shape


# In[ ]:


#np.min(X_train),np.max(X_train,)


# In[ ]:



#X_train,y_train,X_val,y_val,classes=catdog(28)


# In[ ]:



#X_train,y_train,X_val,y_val,classes=Dolphins()


# In[ ]:



#X_train,y_train,X_val,y_val,classes=Pubmed()


# In[ ]:


# tmp=list(map(tuple,y_val))
# A=[]
# print('val')
# for i in set(tmp):
#     A.append( tmp.count(i))
#     print(i, tmp.count(i)/np.sum(tmp))
# tmp=list(map(tuple,y_train))
# B=[]
# print('train')
# for i in set(tmp):
#     B.append( tmp.count(i))
#     print(i, tmp.count(i)/np.sum(tmp))


# In[ ]:


def TASK_1(dataSetName,X_train,y_train,X_val,y_val,classes,layers,words=None,maxEpochs=100,minLayerSize=1500):
    m=X_train.shape[1]
    l=y_train.shape[1]
    results={}
    
    Activators=['sigmoid','tanh','relu','swish']
    
    #sizes=[2**i for i in range( 2*round(np.log2(16)) )  ]
    alphas=[10**(-i) for i in range(0,layers+1)]
    gmi=np.minimum(m//2,minLayerSize)
    HLList=[]
    AList=[]
    LRList=[]
    FULL_TEST=[[],[],[],[],[]]
    counter=1
    for layer in range(1,layers+1):
            tacc=0
            tbstHL=None
            tbstAL=None
            tbstLR=None
            tbstCost=None
            for i in Activators:
                start_time=datetime.datetime.now()
            
                tmpHL=HLList+[(gmi)]
                tmpAL=AList+[(i)]
                tmpLR=LRList+[(alphas[layer])]
                #[gmi,ac,alphas[layer]]   
                print()
                print('tmphl:',tmpHL,'tmpal:',tmpAL,'tmplr:',tmpLR)
                try:
                    net=neuralNetwork(np.array(X_train),np.array(y_train),classes,dataSetName=dataSetName,wInit="he",mode='gaussian' ,hiddenlayers=tmpHL,activations=tmpAL+['soft-max'],cost='L2',learningRate=tmpLR+[0.0003])#mnist
                    costs=(net.train(initADAMS=False,batch_size=1000,doOp=False,epochs=maxEpochs,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,minEpochs=1,patience=0))   
                    path='plots/{0}/TASK1/{0}_{1}.png'.format(dataSetName,counter)
                    counter+=1
                    plotGraph(costs,path,net,plot=False)
                    tmpCost=costs[max(costs)]
                    if(tacc<tmpCost[0]):
                        tbstHL=tmpHL
                        tbstAL=tmpAL
                        tbstLR=tmpLR
                        tacc=tmpCost[0]
                        tbstCost=tmpCost
                    FULL_TEST[layer].append([tmpHL,tmpAL,tmpLR,tmpCost] )
                except:
                    print('Ignoring case:',tmpHL,tmpAL,tmpLR)
                end_time=datetime.datetime.now()
                print('time taken for {0} is {1}'.format(tmpHL,end_time-start_time))
            gmi=int(round((gmi*l)**(0.5)))
            results[layer]=tbstCost
            HLList=tbstHL
            AList=tbstAL
            LRList=tbstLR
    
    return FULL_TEST,results,[HLList,AList,LRList]
    
    


# In[ ]:


def TASK_2(dataSetName,X_train,y_train,X_val,y_val,classes,
                params,layers=3,words=None,maxEpochs=100,minLayerSize=2000):
    m=X_train.shape[1]
    l=y_train.shape[1]
    HLList,AList,LRList=params
    bs=[]
    print(HLList,'\n',AList,'\n',LRList)
    sizes=[2**i for i in range( int(round(np.log2(np.minimum(m,minLayerSize)))))]
    print(sizes)
    bestHL=[]
    bestACC=0
    bestLR=[]
    for hlayer in range(1,layers+1):
        print('Hidden layer:',hlayer)
        al=AList[0:hlayer]
        lr=LRList[0:hlayer]
        tbstHL=[]
        results={}
        tacc=0
        tbstLR=[]
        print('activations:',al)
        for size in sizes:
            tmpHL=bs+[size]
            start_time=datetime.datetime.now()
            
            print('tmphl:',tmpHL)
            try:
                net=neuralNetwork(np.array(X_train),np.array(y_train),classes,dataSetName=dataSetName,wInit="he",mode='gaussian',hiddenlayers=tmpHL,activations=al+['soft-max'],cost='L2',learningRate=lr+[0.0003])#mnist

                #plotGraph(net,path,plot=False)
                costs=(net.train(initADAMS=False,batch_size=1000,doOp=False,epochs=maxEpochs,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,minEpochs=1,patience=0))   
                tmpCost=costs[max(costs)]
                if(tacc<tmpCost[0]):
                    tbstHL=tmpHL
                    tbstLR=lr
                    tacc=tmpCost[0]
                results[size]=tmpCost
            except:
                print('Ignoring case:',tmpHL,al,lr)
            end_time=datetime.datetime.now()
            print('time taken for {0} is {1}'.format(tmpHL,end_time-start_time))
        path='plots/{0}/TASK2/{0}_Layer_{1}.png'.format(dataSetName,hlayer)
        plotGraph(results,path,[dataSetName,'My_NN',2,hlayer],plot=False,Xtitle='neurons')
        bs=tbstHL
        if(bestACC<tacc):
            bestHL=tbstHL
            bestLR=tbstLR
            
    return bestHL
    #Cases=[ [ ['relu'] ]]
    #for element in itertools.product(*Cases):
    #print(element)
    #plotGraph(costs,fig_name,net=None,plot=True,Xtitle='Layer Count'):


# In[ ]:


def TASK_3(dataSetName,X_train,y_train,X_val,y_val,classes,
                HLList,words=None,maxEpochs=100):
    m=X_train.shape[1]
    l=y_train.shape[1]
    hlayers=len(HLList)
    alphas=[10**(-i) for i in range(1,hlayers+1)]
    print('Config:',HLList)
    bestAL=[]
    tacc=0
    Cases=['relu','sigmoid','swish','tanh']
    for element in itertools.product(Cases,repeat=hlayers):
        al=list(element)  
        results={}
        #costs=[]
        #net=None
        start_time=datetime.datetime.now()
        print('activations:',al)
        try:
            net=neuralNetwork(np.array(X_train),np.array(y_train),classes,dataSetName=dataSetName,wInit="he",mode='gaussian',hiddenlayers=HLList,activations=al+['soft-max'],cost='L2',learningRate=alphas+[0.0003])#mnist
            #plotGraph(net,path,plot=False)
            costs=(net.train(initADAMS=False,batch_size=1000,doOp=False,epochs=maxEpochs,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,minEpochs=1,patience=0))   
            tmpCost=costs[max(costs)]

        except:
           print('Ignoring case:')
        end_time=datetime.datetime.now()
        print('time taken for {0} is {1}'.format(al,end_time-start_time))       
        path='plots/{0}/TASK3/{0}_Layer_{1}.png'.format(dataSetName,al)
        plotGraph(costs,path,net,plot=False)
        if(tacc<tmpCost[0]):
            tacc=tmpCost[0]
            bestAL=al
            

    return bestAL
    #Cases=[ [ ['relu'] ]]
    #for element in itertools.product(*Cases):
    #print(element)
    #plotGraph(costs,fig_name,net=None,plot=True,Xtitle='Layer Count'):


# In[ ]:


def Task_4(dataSetName,X_train,y_train,X_val,y_val,classes,
                bestParams,words=None,maxEpochs=100):
    m=X_train.shape[1]
    l=y_train.shape[1]
    al,HLList=bestParams
    hlayers=len(HLList)
    alphas=[10**(-i) for i in range(1,hlayers+1)]
    
    net=neuralNetwork(np.array(X_train),np.array(y_train),classes,dataSetName=dataSetName,wInit=True,mode='gaussian',hiddenlayers=HLList,activations=al+['soft-max'],cost='L2',learningRate=alphas+[0.0003])#mnist
    costs=(net.train(initADAMS=False,batch_size=1000,doOp=False,epochs=maxEpochs,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,minEpochs=1,patience=0))   
            
    path='output_plots/{0}/TASK4/{0}_{1}.png'.format(dataSetName,[net.wInit,net.mode])
        
    plotGraph(costs,path,net,plot=False)
        
    net2=neuralNetwork(np.array(X_train),np.array(y_train),classes,dataSetName=dataSetName,wInit=True,mode='uniform',hiddenlayers=HLList,activations=al+['soft-max'],cost='L2',learningRate=alphas+[0.0003])#mnist
    costs=(net2.train(initADAMS=False,batch_size=1000,doOp=False,epochs=maxEpochs,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,minEpochs=1,patience=0))   
            
    path='output_plots/{0}/TASK4/{0}_{1}.png'.format(dataSetName,[net2.wInit,net2.mode])
    
    plotGraph(costs,path,net2,plot=False)
        
    net3=neuralNetwork(np.array(X_train),np.array(y_train),classes,dataSetName=dataSetName,wInit=False,mode='gaussian',hiddenlayers=HLList,activations=al+['soft-max'],cost='L2',learningRate=alphas+[0.0003])#mnist
    costs=(net3.train(initADAMS=False,batch_size=1000,doOp=False,epochs=maxEpochs,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,minEpochs=1,patience=0))   
            
    path='output_plots/{0}/TASK4/{0}_{1}.png'.format(dataSetName,[net3.wInit,net3.mode])
    
    plotGraph(costs,path,net3,plot=False)
        
    net4=neuralNetwork(np.array(X_train),np.array(y_train),classes,dataSetName=dataSetName,wInit=False,mode='uniform',hiddenlayers=HLList,activations=al+['soft-max'],cost='L2',learningRate=alphas+[0.0003])#mnist
    costs=(net4.train(initADAMS=False,batch_size=1000,doOp=False,epochs=maxEpochs,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,minEpochs=1,patience=0))   
            
    path='output_plots/{0}/TASK4/{0}_{1}.png'.format(dataSetName,[net4.wInit,net4.mode])
    
    plotGraph(costs,path,net4,plot=False)
        


# In[ ]:


def doTASKS(X_train,y_train,X_val,y_val,classes,I=2,dataSetName="cat-dog",mx=20,whichModel="MY_NN"):
    print('doing task1')
    
    full,task1res,bestParams=TASK_1(dataSetName,X_train,y_train,X_val,y_val,classes,3,words=None,maxEpochs=mx)

    fig_name='output_plots/{0}/{0}_{1}.png'.format(dataSetName,'RESULT')
    plotGraph(task1res,fig_name,[dataSetName,whichModel,I,1],plot=False)
    #'DataSet={0}, model={1}, part={2}, task={3}'
    #print('hello worlds')
    print('Best parameter from task1s:',bestParams)
    print('Doing task2')
    bestHL=TASK_2(dataSetName,X_train,y_train,X_val,y_val,classes,bestParams,3,words=None,maxEpochs=mx)
    print('Best Hiddenlayer from task2s:',bestHL)
    print('Doing task3')
    bestACC=TASK_3(dataSetName,X_train,y_train,X_val,y_val,classes,bestHL,words=None,maxEpochs=mx)
    print('bestACC,bestHL',bestACC,bestHL)  
    print('Doing Task4')
    Task_4(dataSetName,X_train,y_train,X_val,y_val,classes,[bestACC,bestHL],words=None,maxEpochs=mx)
    print('MY_NN tasks Complete.')


# In[ ]:


#doTASKS(X_train,y_train,X_val,y_val,classes,2,"cat-dog")


# In[ ]:


#Task_4("cat-dog",X_train,y_train,X_val,y_val,classes,[['relu', 'swish', 'swish'],[512, 128, 128]],words=None,maxEpochs=20)


# Swish Function to Keras

# In[ ]:


# Ref: https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def swish2(x):
    return x*K.sigmoid(x)

get_custom_objects().update({'swish': Activation(swish2)})

def addswish(model):
    model.add(Activation(swish2))


# In[ ]:


def task5(X_train,y_train,X_val,y_val,classes,HLList,ALList):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    m=X_train.shape[1]
    l=y_train.shape[1]
    
    model.add(Dense(HLList[0], activation=ALList[0], input_dim=m))
    model.add(Dropout(0.5))
    flag=True
    for x,y in zip(HLList,ALList):
        if(flag):
            flag=False
            continue
        model.add(Dense(x, activation=y))
        model.add(Dropout(0.5))
        
    model.add(Dense(l, activation='softmax'))

    sgd = SGD(lr=0.01, decay=0, momentum=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              epochs=20,
              batch_size=128)
    y_pred = model.predict(X_val)
    #y_pred=oneHot(y_pred,classes)
    #scoreTrain=model.evaluate(X_train, y_train, batch_size=128)
    #scoreVal = model.evaluate(X_val, y_val, batch_size=128)
    #print(scoreTrain,scoreVal)
    y_val=np.argmax(y_val,axis=1)
    y_pred=np.argmax(y_pred,axis=1)
    #print(y_val,y_pred)
    print(np.mean(y_val==y_pred))


# In[ ]:


#bestParams,bestACC,bestHL
#task5(X_train,y_train,X_val,y_val,classes,[512, 128, 128],['relu', 'swish', 'swish'])


# In[ ]:


#np.save('cat-dog_MODEL/full_res',[full,res])


# In[ ]:


#fullres=np.load('cat-dog_MODEL/full_res.npy').tolist()


# In[ ]:


# dataSetName="MNIST"
# fig_name='plots/{0}/{0}_{1}.png'.format(dataSetName,'RESULT')
# plotGraph(task1res,fig_name,[dataSetName,'My_NN',1,1],plot=False)
# #'DataSet={0}, model={1}, part={2}, task={3}'
# #print('hello worlds')


# In[ ]:


#task1res


# In[ ]:


#Twitter
# X_train,y_train,X_val,y_val,classes,words=Twitter()
# X_train.shape,y_train.shape, X_val.shape, y_val.shape
# net,X_val,y_val=getNET(X_train,y_train,X_val,y_val,classes,words=None)
# costs=[]
# costs=(net.train(initADAMS=True,batch_size=1000,doOp=True,epochs=15,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,minEpochs=1,patience=0))  


# In[ ]:


#updates
def Twitter():
    X=np.array(LoadDataForTXT('D:/workspace/tipr/tipr 2nd ass/tipr-second-assignment/data/twitter/twitter.txt'))
    y=np.array(LoadDataForTXT('D:/workspace/tipr/tipr 2nd ass/tipr-second-assignment/data/twitter/twitter_label.txt'))
    y=y.T[0]
    X,words=BagOfWords(X,keys=None)
    X,y=shuffle(X,y)
    X,y,classes=preprocess(X,y,"Twitter")
    X_train,y_train,X_val,y_val=train_test_split(X,y,.2)
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_val=np.array(X_val)
    y_val=np.array(y_val)
    return X_train,y_train,X_val,y_val,classes,words
# def getNET(X_train,y_train,X_val,y_val,classes,words):
#     gm1=X_train.shape[1]*2
#     gm2=round((gm1*y_train.shape[1])**(0.5))
#     gm3=round((gm2*y_train.shape[1])**(0.5)) 
#     myList=np.array([[gm1,'relu',0.1],[gm2,'tanh',0.01]])
#     net=neuralNetwork(X_train,y_train,classes,dataSetName="Twitter",hiddenlayers=[gm1,gm2],activations=['relu','swish','soft-max'],cost='cross_entropy',learningRate=[.3,.03,0.0003])#mnist
#     net.layers
#     net.classes
#     net.words=words
#     return net,X_val,y_val


# In[ ]:


def dummyDataSet(dataname):
    X,y=sklearn.datasets.load_digits(n_class=10, return_X_y=True)
    from sklearn.utils import shuffle
    X,y=shuffle(X,y,random_state=26)

    X,y,classes=preprocess(X,y,dataname)
    X_train,y_train,X_val,y_val=train_test_split(X,y)
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_val=np.array(X_val)
    y_val=np.array(y_val)
    #return X_train,y_train,X_val,y_val

    #X,y=sklearn.datasets.load_iris(return_X_y=True)
    gm1=X.shape[1]*2
    gm2=int((gm1*10)**(0.5))
#X_train=np.copy(X)
#y_train=np.copy(y)
#classes=[i for i in range(0,10)]
    print(X_train.shape)
    net=neuralNetwork(X_train,y_train,classes,dataSetName="MNIST",hiddenlayers=[gm1,gm2],activations=['relu','tanh','soft-max'],cost='L2',learningRate=[.1,.001,.0001])#mnist
    #net=neuralNetwork(X,y.reshape(-1,1),hiddenlayers=[gm1],activations=['tanh','soft-max'],cost='L2',learningRate=[.1,.001])#iris

    #net.y,net.classes=net.oneHot(y)
    
    net.layers
    net.classes
    #costs=net.train(epochs=1000)
    #costs=net.train(batch_size=2,epochs=100,KKK=10,earlyStopping=True,X_val=X_val,y_val=y_val)  
    #fig_name='myPlot2.png'
    #plotGraph(net,costs,fig_name)
    return net,X_val,y_val
    net,X_val,y_val=dummyDataSet()


# In[ ]:


def catdog(model_path,path='D:/workspace/tipr/tipr 2nd ass/tipr-second-assignment/data/MNIST',IMG_SIZE=28):
    X=np.array(loadDataSet2('{0}/{1}'.format(path,"cat") ,itr=None,IMG_SIZE=IMG_SIZE,as_gray=True))
    y=[0]*X.shape[0]
    
    X=X.reshape(X.shape[0],-1)
    print('cat',X.shape)
    #X=scale(X)
    for i in range(1,2):
        tmp_X=np.array(loadDataSet2('{0}/{1}'.format(path,"dog"),itr=None,IMG_SIZE=IMG_SIZE,as_gray=True))
        tmp_y=[i]*tmp_X.shape[0]
        print(tmp_X.shape)
        tmp_X=tmp_X.reshape(tmp_X.shape[0],-1)
        #tmp_X=scale(tmp_X)
       
        X=np.append(X,tmp_X,axis=0)
        y=np.append(y,tmp_y)
        print(X.shape,len(y))
    X,y=shuffle(X,y)
    X,y,classes=preprocess(X,y,"cat-dog",model_path,doScale=True)
    X_train,y_train,X_val,y_val=train_test_split(X,y,.2)
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_val=np.array(X_val)
    y_val=np.array(y_val)
    return X_train,y_train,X_val,y_val,classes
# def getNET(X_train,y_train,X_val,y_val,classes):
#     gm1=X_train.shape[1]//2
#     gm2=round((gm1*y_train.shape[1])**(0.5))*4
#     gm3=round((gm2*y_train.shape[1])**(0.5)) 
#     myList=np.array([[gm1,'relu',0.1],[gm2,'tanh',0.01]])
#     net=neuralNetwork(X_train,y_train,classes,dataSetName="CAT_DOG",hiddenlayers=[gm1,gm2],activations=['relu','tanh','soft-max'],cost='cross_entropy',learningRate=[.3,.01,.0001])#mnist
#     net.layers
#     net.classes
#     return net,X_val,y_val


# In[ ]:


#updates
def Dolphins():
    X=np.array(LoadDataForCSV('D:/workspace/tipr/tipr 2nd ass/tipr-second-assignment/data/dolphins/dolphins.csv'))
    y=np.array(LoadDataForCSV('D:/workspace/tipr/tipr 2nd ass/tipr-second-assignment/data/dolphins/dolphins_label.csv'))
    y=y.T[0]
    X,y=shuffle(X,y)
    X,y,classes=preprocess(X,y,"Dolphins")
    X_train,y_train,X_val,y_val=train_test_split(X,y,.2)
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_val=np.array(X_val)
    y_val=np.array(y_val)
    return X_train,y_train,X_val,y_val,classes
# def getNET(X_train,y_train,X_val,y_val,classes):
#     gm1=X_train.shape[1]*2
#     gm2=round((gm1*y_train.shape[1])**(0.5))
#     gm3=round((gm2*y_train.shape[1])**(0.5)) 
#     myList=np.array([[gm1,'relu',0.1],[gm2,'tanh',0.01]])
#     net=neuralNetwork(X_train,y_train,classes,dataSetName="dolphins",hiddenlayers=[gm1,gm2],activations=['tanh','tanh','soft-max'],cost='cross_entropy',learningRate=[.1,.1,0.1])#mnist
#     net.layers
#     net.classes
#     return net,X_val,y_val


# In[ ]:


#updates
def Pubmed():
    X=np.array(LoadDataForCSV('tipr-second-assignment/data/pubmed/pubmed.csv'))
    y=np.array(LoadDataForCSV('tipr-second-assignment/data/pubmed/pubmed_label.csv'))
    y=y.T[0]
    X,y=shuffle(X,y)
    X,y,classes=preprocess(X,y,"Pubmed")
    X_train,y_train,X_val,y_val=train_test_split(X,y,.2)
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_val=np.array(X_val)
    y_val=np.array(y_val)
    return X_train,y_train,X_val,y_val,classes
# def getNET(X_train,y_train,X_val,y_val,classes):
#     gm1=X_train.shape[1]*2
#     gm2=round((gm1*y_train.shape[1])**(0.5))
#     gm3=round((gm2*y_train.shape[1])**(0.5)) 
#     myList=np.array([[gm1,'relu',0.1],[gm2,'tanh',0.01]])
#     net=neuralNetwork(X_train,y_train,classes,dataSetName="pubmed",hiddenlayers=[gm1,gm2],activations=['relu','tanh','soft-max'],cost='cross_entropy',learningRate=[.1,.01,0.01])#mnist
#     net.layers
#     net.classes
#     return net,X_val,y_val


# In[ ]:


def MNIST(model_path,path='D:/workspace/tipr/tipr 2nd ass/tipr-second-assignment/data/MNIST'):
    X=np.array(loadDataSet('{0}/{1}'.format(path,0)))
    y=[0]*X.shape[0]
    for i in range(1,10):
        tmp_X=np.array(loadDataSet('{0}/{1}'.format(path,i)))
        tmp_y=[i]*tmp_X.shape[0]
        print(tmp_X.shape)
        X=np.append(X,tmp_X,axis=0)
        y=np.append(y,tmp_y)
        print(X.shape,len(y))
    X=X.reshape(X.shape[0],-1)
    X,y=shuffle(X,y)
    X,y,classes=preprocess(X,y,"MNIST",model_path,doScale=True)
    X_train,y_train,X_val,y_val=train_test_split(X,y,.2)
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_val=np.array(X_val)
    y_val=np.array(y_val)
    return X_train,y_train,X_val,y_val,classes
# def getNET(X_train,y_train,X_val,y_val,classes):
#     gm1=X_train.shape[1]*2
#     gm2=round((gm1*y_train.shape[1])**(0.5))
#     gm3=round((gm2*y_train.shape[1])**(0.5)) 
#     myList=np.array([[gm1,'relu',0.1],[gm2,'tanh',0.01]])
#     net=neuralNetwork(X_train,y_train,classes,dataSetName="MNIST",hiddenlayers=[gm1,gm2],activations=['tanh','sigmoid','soft-max'],cost='L2',learningRate=[0.3,0.003])#mnist
#     net.layers
#     net.classes
#     return net,X_val,y_val


# In[ ]:


def TestMNIST(test_path,model_path="MNIST_MODEL/"):
        X=np.array(loadDataSet('{0}/{1}'.format(test_path,0)))
        y=[0]*X.shape[0]
        for i in range(1,10):
            tmp_X=np.array(loadDataSet('{0}/{1}'.format(test_path,i)))
            tmp_y=[i]*tmp_X.shape[0]
            #print(tmp_X.shape)
            X=np.append(X,tmp_X,axis=0)
            y=np.append(y,tmp_y)
            #print(X.shape,len(y))
        X=X.reshape(X.shape[0],-1)
        X_,y_,class_=preprocess(X,y,"MNIST",model_path,doScale=True,testing=False)
       
        net,X_,y_=getNET(X,y_,X,y_,"MNIST",[5,2],['relu','tanh'])
        net.loadModel(model_path+"/Model_Main.npy")
        X,y=preprocess(X,y,"MNIST",model_path,doScale=True,testing=True,classes=net.classes)
       
        net.testModel(X,y)


# In[ ]:


#tstpath='D:/workspace/tipr/tipr 2nd ass/tipr-second-assignment/data/MNIST'
#TestMNIST(tstpath)


# In[ ]:


def Testcatdog(test_path,model_path="cat-dog_MODEL",IMG_SIZE=28):
        X=np.array(loadDataSet2('{0}/{1}'.format(test_path,"cat") ,itr=None,IMG_SIZE=IMG_SIZE,as_gray=True))
        y=[0]*X.shape[0]

        X=X.reshape(X.shape[0],-1)
        print('cat',X.shape)
        #X=scale(X)
        for i in range(1,2):
            tmp_X=np.array(loadDataSet2('{0}/{1}'.format(test_path,"dog"),itr=None,IMG_SIZE=IMG_SIZE,as_gray=True))
            tmp_y=[i]*tmp_X.shape[0]
            print(tmp_X.shape)
            tmp_X=tmp_X.reshape(tmp_X.shape[0],-1)
            #tmp_X=scale(tmp_X)

            X=np.append(X,tmp_X,axis=0)
            y=np.append(y,tmp_y)
            print(X.shape,len(y))
            
        X_,y_,class_=preprocess(X,y,"cat-dog",model_path,doScale=True,testing=False)
       
        net,X_,y_=getNET(X,y_,X,y_,"cat-dog",[5,2],['relu','tanh'])
        net.loadModel(model_path+"/Model_Main.npy")
        X,y=preprocess(X,y,"cat-dog",model_path,doScale=True,testing=True,classes=net.classes)
       
        net.testModel(X,y)


# In[ ]:


#tstpath='D:/workspace/tipr/tipr 2nd ass/tipr-second-assignment/data/cat-dog'
#Testcatdog(tstpath)


# In[ ]:


def getNET(X_train,y_train,X_val,y_val,dataSetName="Twitter",HLList=[1568,128],activations=['relu','tanh'],classes=None,words=None):
    alphas=[10**(-i) for i in range(1,len(HLList)+1)]
    lr=alphas 
    net=neuralNetwork(X_train,y_train,classes,dataSetName=dataSetName,hiddenlayers=HLList,activations=activations+['soft-max'],cost='L2',learningRate=lr+[10**(-(len(HLList)+1)) ])#mnist
    net.layers
    net.classes
    net.words=words
    return net,X_val,y_val


# In[ ]:


#net,X_val,y_val=getNET(X_train,y_train,X_val,y_val,"cat-dog",[512, 128, 128],['relu', 'swish', 'swish'],classes)
#costs=(net.train(initADAMS=False,batch_size=1000,doOp=False,epochs=20,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,printResults=True,minEpochs=1,patience=0))  


# In[ ]:


#y_val


# In[ ]:


#net.learningRate=np.array([0.03]*2)
#net.testModel(X_val,y_val)


# In[ ]:


#net.learningRate[0]=0.3
#net.learningRate


# In[ ]:


#costs=(net.train(initADAMS=False,batch_size=1000,doOp=False,epochs=20,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,printResults=True,minEpochs=1,patience=0))  


# In[ ]:


#costs


# In[ ]:


#net.loadModel("cat-dog_MODEL/Model_Main.npy")


# In[ ]:


#net.loadModel("xxx.npy")


# In[ ]:


# net.dataSetName="cat-dog"
# fig_name='plots/{0}/{0}_{1}.png'.format(net.dataSetName,'9')
# plotGraph(costs,fig_name,net)


# In[ ]:


# X_train[0].reshape(100,100)


# In[ ]:


# len(costs)


# In[ ]:


# cv2.imshow('image',X_train[0].reshape(100,100))


# In[ ]:


# [np.array(i).shape for i in net.bias]


# In[1]:


# [np.min(i) for i in net.bias[1:]]


# So far For Mnist Dataset.
# Config is:-<br>
# <b>net=neuralNetwork(X,y.reshape(-1,1),hiddenlayers=[128,gm],activations=['relu','tanh','soft-max'],cost='L2',learningRate=[.01,.01,.01])</b>  acc = 99.99% <br><br>
# net=neuralNetwork(X,y.reshape(-1,1),hiddenlayers=[128,gm],activations=['relu','relu','soft-max'],cost='L2',learningRate=[.01,.01,.01]) acc= 99% but chokes for NAN<br>
# <br>For Scale :-
# Normalize is set to false for both case. Min-max works better. with norm then minmax, cost goes down but slower than just minmax.
# <br>Seed set to 26.
# 
# 

# In[ ]:


argv = list(sys.argv)
dataSetName = argv[argv.index('--dataset')+1]
test_path = argv[argv.index('--test-data')+1]
if '--train-data' in argv:
    train_path = argv[argv.index('--train-data')+1]
    config   = []#HiddenLayers for now.
    model_path="temp"
    for st in argv[argv.index('--configuration')+1:]:
        st  = st.strip()
        if st.endswith(']'):
            config.append(int(st.strip('[]')))
            break
        else:
            config.append(int(st.strip('[]')))
    if(dataSetName=="MNIST"):
        X_train,y_train,X_val,y_val,classes=MNIST(train_path)
        net,X_val,y_val=getNET(X_train,y_train,X_val,y_val,dataSetName,config,['relu', 'swish', 'swish'][:len(config)],classes)
    elif(dataSetName=="Cat-Dog" or dataSetName=="cat-dog" ):
        X_train,y_train,X_val,y_val,classes=catdog(28,train_path)
        net,X_val,y_val=getNET(X_train,y_train,X_val,y_val,dataSetName,config,['relu', 'swish', 'swish'][:len(config)],classes)
        
    net.train(initADAMS=False,batch_size=1000,doOp=False,epochs=50,KKK=1,earlyStopping=True,X_val=X_val,y_val=y_val,printResults=True,minEpochs=1,patience=0)  
    net.saveModel("temp/{0}".format("Model_Main"))
    model_name="{0}/Model_Main.npy".format(model_path)
    
else:
    if(dataSetName=="MNIST"):
           model_path="MNIST_model/"

    elif(dataSetName=="Cat-Dog" or dataSetName=="cat-dog" ):
           model_path="cat-dog_model/"
print('Testing Model',test_path,model_path)
if(dataSetName=="MNIST"):
    TestMNIST(test_path,model_path)
elif(dataSetName=="Cat-Dog" or dataSetName=="cat-dog" ):
     Testcatdog(test_path,model_path) 

