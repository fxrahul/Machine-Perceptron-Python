# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:49:32 2019

@author: 91755
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:18:10 2019

@author: 91755
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:00:57 2019

@author: 91755
"""
import numpy as np 
import pandas as pd



class PerceptronTraining():
    def __init__(self):
       np.random.seed(1)
       self.weights = 2* np.random.random((2,1)) - 1
    
    def sig(self, x):
        calculate = 1/(1+np.exp(-x))
        calculate[calculate >= 0.5] = 1
        calculate[calculate < 0.5] = 0
        return calculate
    
    def sig_derivative(self,x):
        cal = x * (1-x)
        return cal
    
    def training_time(self, train_inputs, train_outputs, no_of_epoch):
        
        for i in range(no_of_epoch):
            pred_output = self.predValue(train_inputs)
            error = train_outputs - pred_output
            ajustmentForWeights = np.dot(train_inputs.T,0.001*error*self.sig_derivative(pred_output))
            self.weights += ajustmentForWeights
              
            
        
            
            
    def predValue(self,inputs):
        inputs = inputs.astype(float)
        output = self.sig(np.dot(inputs, self.weights))
        return output
    
    def checkLabel(self,inp):
        if inp == '+':
            return '1'
        else:
            return '-1'
    
    def predAccuracy(self,originalLabel,predValue):

        matched = 0
        for i in range(len(originalLabel)):
                if originalLabel[i] == predValue[i]:
                    matched += 1
        accuracyVal = matched / float(len(originalLabel)) * 100.0       
        return accuracyVal
   

if __name__ == "__main__":
    rad =2
    d =0
    n_samp =1000
    width =3
    if rad < width/2:
       print('The radius should be at least larger than half the width')
        
    if n_samp % 2 != 0 :
       print('Please make sure the number of samples is even')
            
    aa= np.random.rand(2,(int)(n_samp/2))
    radius = (rad-width/2) + width*aa[0,:]
    radius=np.reshape(radius, (1,np.product(radius.shape))) 
    theta = 3.14*aa[1,:]
    theta=np.reshape(theta, (1,np.product(theta.shape))) 
        
        
    x  = radius*np.cos(theta)
    x=np.reshape(x, (1,np.product(x.shape))) 
    y  = radius*np.sin(theta)
    y=np.reshape(y, (1,np.product(y.shape))) 
    label = 1*np.ones([1,x.size])
        
    x1  = radius*np.cos(-theta)+rad
    x1=np.reshape(x1, (1,np.product(x1.shape))) 
    y1  = radius*np.sin(-theta)-d
    y1=np.reshape(y1, (1,np.product(y1.shape))) 
    label1 = 0*np.ones([1,x.size])
        
        
    data1 = np.vstack(( np.hstack((x,x1)),np.hstack((y,y1)) ))
    data2 = np.hstack( (label,label1) )
    data = np.concatenate( (data1,data2 ),axis=0 )
    n_row = data.shape[0]
    n_col = data.shape[1]
    shuffle_seq = np.random.permutation(n_col)
        
        
    data_shuffled = np. random.rand(3,1000)
    for i in range(n_col):
       data_shuffled[:,i] = data[:,shuffle_seq[i] ];
            
        #print(data_shuffled[0] [0])
        #print(data_shuffled[0].shape)
        
        
    train_input_data = np.stack([data_shuffled[0], data_shuffled[1]], axis=1)
   
    nn = PerceptronTraining()
    
    outputLabel  = data_shuffled[2].reshape(1000,1)
    nn.training_time(train_input_data,outputLabel,1000)
    
    rad =2
    d =0
    n_samp =2000
    width =3
    if rad < width/2:
       print('The radius should be at least larger than half the width')
        
    if n_samp % 2 != 0 :
       print('Please make sure the number of samples is even')
            
    aa= np.random.rand(2,(int)(n_samp/2))
    radius = (rad-width/2) + width*aa[0,:]
    radius=np.reshape(radius, (1,np.product(radius.shape))) 
    theta = 3.14*aa[1,:]
    theta=np.reshape(theta, (1,np.product(theta.shape))) 
        
        
    x  = radius*np.cos(theta)
    x=np.reshape(x, (1,np.product(x.shape))) 
    y  = radius*np.sin(theta)
    y=np.reshape(y, (1,np.product(y.shape))) 
    label = 1*np.ones([1,x.size])
        
    x1  = radius*np.cos(-theta)+rad
    x1=np.reshape(x1, (1,np.product(x1.shape))) 
    y1  = radius*np.sin(-theta)-d
    y1=np.reshape(y1, (1,np.product(y1.shape))) 
    label1 = 0*np.ones([1,x.size])
        
        
    data1 = np.vstack(( np.hstack((x,x1)),np.hstack((y,y1)) ))
    data2 = np.hstack( (label,label1) )
    data = np.concatenate( (data1,data2 ),axis=0 )
    n_row = data.shape[0]
    n_col = data.shape[1]
    shuffle_seq = np.random.permutation(n_col)
        
        
    data_shuffled = np. random.rand(3,2000)
    for i in range(n_col):
       data_shuffled[:,i] = data[:,shuffle_seq[i] ];
      
    test_input_data = np.stack([data_shuffled[0], data_shuffled[1]], axis=1)
    pred_test_data_output = nn.predValue(test_input_data)
    actual_test_data_output = data_shuffled[2].reshape(2000,1)
    error_rate = np.square(np.subtract(actual_test_data_output,pred_test_data_output)).mean()
    #accuracy = accuracy_score(pred_test_data_output,actual_test_data_output)
    #print("Accuracy :",accuracy)
    print("Moon Test Data Error Rate :",error_rate*100,"%")
    
    #UCI Dataset
    
   
    # Read data from file 'filename.csv' 
    # (in the same directory that your python process is based)
    # Control delimiters, rows, column names with read_csv (see later) 
    data = pd.read_csv("cc.csv") 
    # Preview the first 5 lines of the loaded data 
    data.head()
    inputData = pd.DataFrame(data, columns = ['a7', 'a10']).to_numpy()
    outLabelWithoutBinary = pd.DataFrame(data, columns = ['a15']).to_numpy()
    
    outLabelWithoutBinary[outLabelWithoutBinary == '+'] = 1
    outLabelWithoutBinary[outLabelWithoutBinary == '-'] = 0

    
        
    uciOutput = nn.predValue(inputData)
    
    #error_rate = np.square(np.subtract(uciOutput,outLabelWithoutBinary)).mean()
    #acc = jaccard_score(outLabelWithoutBinary, uciOutput) 
    accuracy = nn.predAccuracy(outLabelWithoutBinary,uciOutput)
    #print("Accuracy :",accuracy)
    #print("UCI Test Data Error :",error_rate)
    print("UCI Dataset Accuracy :",accuracy, "%")
    
    
    
    
                       
                           
    #print(outLabelWithoutBinary)
    

		  
        
   

    

 
   