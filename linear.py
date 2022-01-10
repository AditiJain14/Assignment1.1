#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[1]:


import sys
#no of args=5 part a
#no of args=7 part b
#no of args=4 part c


# In[ ]:


def LR_A(train,test_x,output_file,output_weight):
    Y=train["Total Costs"]
    del train["Total Costs"]
    X=train.to_numpy()
    X=np.c_[np.ones(train.shape[0]),X]
    y=Y.to_numpy()
    trans=X.transpose()
    dot=np.dot(trans,X)
    inv=np.linalg.inv(dot)
    W=np.matrix(np.dot(inv,np.dot(trans,y)))
    W=W.transpose()
    test_x=test_x.to_numpy()
    test_x=np.c_[np.ones(test_x.shape[0]),test_x]
    y_test=np.dot(test_x,W)
    y_test=y_test.flatten()
    saved_data_test = list(zip(y_test))
    df = pd.DataFrame(np.transpose(saved_data_test[0][0][0][0][0]))
    df.to_csv(output_file,index = False, header=False)
    #f = open(output_file, "w")
    W=W.flatten()
    saved_data = list(zip(W))
    df = pd.DataFrame(np.transpose(saved_data[0][0][0][0][0]))
    df.to_csv(output_weight,index = False,header=False)
    return output_file,output_weight


# In[ ]:


def LR_B(train,test_x,lambd,output_file,output_weight,best_lambda):
    Y=train["Total Costs"]
    del train["Total Costs"]
    X=train.to_numpy()
    X=np.c_[np.ones(train.shape[0]),X]
    y=Y.to_numpy()
    def find_weights(X,Y,X_test,lamb):
        trans=X.transpose()
        iden=np.identity(X.shape[1])
        dot=np.dot(trans,X)+iden*lamb
        inv=np.linalg.inv(dot)
        W=np.matrix(np.dot(inv,np.dot(trans,Y)))
        W=W.transpose()
        y_test=np.dot(X_test,W)
        return y_test,W
    def test_train_split(X,k,i,Y):
        size=X.shape[0]//k
        end=(i+1)*size
        start=i*size
        if(i==0):
            X_train=X[end:X.shape[0]]
            Y_train=Y[end:X.shape[0]]
        else:
            X_train=X[0:start]
            Y_train=Y[0:start]
            X_train=np.r_[X_train,X[end:X.shape[0]]]
            Y_train=np.r_[Y_train,Y[end:X.shape[0]]]
        X_t=X[start:end]
        Y_t=Y[start:end]
        return X_train,X_t,Y_train,Y_t
    def loss(Y_t,Y_p,testsize):
        los=0
        Y_t=np.reshape(Y_t,(testsize,1))
        diff=np.subtract(Y_t,Y_p)
        diff=np.dot(np.transpose(diff),diff)
        div=np.dot(np.transpose(Y_t),Y_t)
        los=np.sqrt(diff/div)
        return los
    def cross_validation(lambd,X,Y,k):
        mean=np.zeros((lambd.size))
        l=0
        for i in range(0,lambd.shape[1]):
            j=lambd[:,i]
            loss_Sum=0
            for i in range(0,k):
                x_train,x_test,y_train,y_test=test_train_split(X,k,i,Y)
                y_predict,W=find_weights(x_train,y_train,x_test,j)
                total=k*x_test.size
                testsize=y_test.size
                lo=loss(y_test,y_predict,testsize)
                loss_Sum=loss_Sum+lo
            mean[l]=loss_Sum/total
            l=l+1
        return mean
    
    mean=cross_validation(lambd,X,y,10)
    minimum=min(mean)
    i=0
    while(mean[i]!=minimum):
        i=i+1
    lamb=lambd[:,i]
    test_x=test_x.to_numpy()
    test_x=np.c_[np.ones(test_x.shape[0]),test_x]
    y_test,W=find_weights(X,Y,test_x,lamb)
    y_test=y_test.flatten()
    saved_data_test = list(zip(y_test))
    df = pd.DataFrame(np.transpose(saved_data_test[0][0][0][0][0]))
    df.to_csv(output_file,index = False, header=False)
    W=W.flatten()
    saved_data = list(zip(W))
    df = pd.DataFrame(np.transpose(saved_data[0][0][0][0][0]))
    df.to_csv(output_weight,index = False,header=False)
    saved_lamb=list(zip(lamb))
    df=pd.DataFrame(saved_lamb)
    df.to_csv(best_lambda,index=False,header=False)
    return output_file,output_weight,best_lambda


# In[ ]:


def LR_C(train_Data,test_Data,outputfile):
    Y=train_Data["Total Costs"]
    del train_Data["Total Costs"]

    def predrop(train_Data):
        del train_Data["Operating Certificate Number"]
        del train_Data["Facility Id"]
        del train_Data["Zip Code - 3 digits"]
        del train_Data["Gender"]
        del train_Data["Race"]
        del train_Data["Ethnicity"]
        del train_Data["CCS Diagnosis Code"]
        del train_Data["CCS Diagnosis Description"]
        del train_Data["CCS Procedure Code"]
        del train_Data["CCS Procedure Description"]
        del train_Data["APR DRG Code"]
        del train_Data["APR Severity of Illness Code"]
        del train_Data["APR DRG Description"]
        del train_Data["APR MDC Code"]
        del train_Data["APR Risk of Mortality"]
        del train_Data["Birth Weight"]
        del train_Data["Payment Typology 3"]
        del train_Data["Emergency Department Indicator"]
#payment 3 added
        return train_Data
    train_Data=predrop(train_Data)
    test_Data=predrop(test_Data)
    def drop(one_hot_encoded_data):
        one_hot_encoded_data.drop(columns=['Health Service Area_1',
 'Health Service Area_2',
 'Health Service Area_7',
 'Health Service Area_8',
 'Hospital County_1',
 'Hospital County_2',
 'Hospital County_4',
 'Hospital County_5',
 'Hospital County_6',
 'Hospital County_7',
 'Hospital County_9',
 'Hospital County_10',
 'Hospital County_11',
 'Hospital County_12',
 'Hospital County_14',
 'Hospital County_15',
 'Hospital County_16',
 'Hospital County_17',
 'Hospital County_18',
 'Hospital County_20',
 'Hospital County_21',
 'Hospital County_22',
 'Hospital County_23',
 'Hospital County_24',
 'Hospital County_25',
 'Hospital County_27',
 'Hospital County_28',
 'Hospital County_31',
 'Hospital County_32',
 'Hospital County_34',
 'Hospital County_37',
 'Hospital County_38',
 'Hospital County_39',
 'Hospital County_40',
 'Hospital County_42',
 'Hospital County_44',
 'Hospital County_45',
 'Hospital County_46',
 'Hospital County_47',
 'Hospital County_48',
 'Hospital County_49',
 'Hospital County_50',
 'Hospital County_51',
 'Hospital County_52',
 'Hospital County_53',
 'Hospital County_56',
 'Hospital County_57',
 'Facility Name_2',
 'Facility Name_3',
 'Facility Name_4',
 'Facility Name_6',
 'Facility Name_17',
 'Facility Name_20',
 'Facility Name_21',
 'Facility Name_23',
 'Facility Name_24',
 'Facility Name_25',
 'Facility Name_29',
 'Facility Name_30',
 'Facility Name_31',
 'Facility Name_32',
 'Facility Name_33',
 'Facility Name_34',
 'Facility Name_36',
 'Facility Name_37',
 'Facility Name_38',
 'Facility Name_39',
 'Facility Name_42',
 'Facility Name_44',
 'Facility Name_46',
 'Facility Name_47',
 'Facility Name_50',
 'Facility Name_51',
 'Facility Name_53',
 'Facility Name_55',
 'Facility Name_56',
 'Facility Name_65',
 'Facility Name_68',
 'Facility Name_69',
 'Facility Name_74',
 'Facility Name_75',
 'Facility Name_80',
 'Facility Name_81',
 'Facility Name_82',
 'Facility Name_83',
 'Facility Name_84',
 'Facility Name_88',
 'Facility Name_92',
 'Facility Name_93',
 'Facility Name_98',
 'Facility Name_101',
 'Facility Name_105',
 'Facility Name_110',
 'Facility Name_113',
 'Facility Name_118',
 'Facility Name_119',
 'Facility Name_124',
 'Facility Name_125',
 'Facility Name_128',
 'Facility Name_129',
 'Facility Name_131',
 'Facility Name_133',
 'Facility Name_137',
 'Facility Name_138',
 'Facility Name_144',
 'Facility Name_146',
 'Facility Name_155',
 'Facility Name_156',
 'Facility Name_157',
 'Facility Name_160',
 'Facility Name_163',
 'Facility Name_165',
 'Facility Name_173',
 'Facility Name_174',
 'Facility Name_178',
 'Facility Name_179',
 'Facility Name_180',
 'Facility Name_183',
 'Facility Name_187',
 'Facility Name_188',
 'Facility Name_192',
 'Facility Name_194',
 'Facility Name_196',
 'Facility Name_203',
 'Facility Name_204',
 'Facility Name_212',
 'Age Group_3',
 'Type of Admission_2',
 'Type of Admission_4',
 'Patient Disposition_1',
 'Patient Disposition_4',
 'Patient Disposition_6',
 'Patient Disposition_7',
 'Patient Disposition_9',
 'Patient Disposition_10',
 'Patient Disposition_12',
 'Patient Disposition_15',
 'APR MDC Description_5',
 'APR MDC Description_8',
 'APR MDC Description_10',
 'APR MDC Description_14',
 'APR Severity of Illness Description_4',
 'APR Medical Surgical Description_1',
 'Payment Typology 1_2',
 'Payment Typology 1_3',
 'Payment Typology 1_4',
 'Payment Typology 2_2',
 'Payment Typology 2_7',
 'Payment Typology 2_8'], axis=1, inplace=True)
        return one_hot_encoded_data
    def feature(train_Data):
        X=train_Data.to_numpy()
        length_s=X[:,train_Data.columns.get_loc('Length of Stay')].reshape(X.shape[0],1)
        colmn=[]
        for i in train_Data.columns:
            if i!="Length of Stay":
                colmn+=[i]
        ''' fake_data=np.zeros((212,13))
        max_hsa=8
        max_hc=57
        max_fcn=212
        max_ag=5
        max_typa=6
        max_patd=19
        max_mdc=24
        max_sevc=4
        max_sevd=4
        max_mor=4
        max_msd=2
        max_ptyp1=10
        max_ptyp2=11
        for i in range(0,212):
            fake_data[i,0]=min(i,max_hsa)
            fake_data[i,1]=min(i,max_hc)
            fake_data[i,2]=min(i,max_fcn)
            fake_data[i,3]=min(i,max_ag)
            fake_data[i,4]=min(i,max_typa)
            fake_data[i,5]=min(i,max_patd)
            fake_data[i,6]=min(i,max_mdc)
            fake_data[i,7]=min(i,max_sevc)
            fake_data[i,8]=min(i,max_sevd)
            fake_data[i,9]=min(i,max_mor)
            fake_data[i,10]=min(i,max_msd)
            fake_data[i,11]=min(i,max_ptyp1)
            fake_data[i,12]=min(i,max_ptyp2)
        df = pd.DataFrame(data = fake_data, 
                  columns =colmn)
        train_Data=train_Data.append(df)'''
        
        one_hot_encoded_data = pd.get_dummies(train_Data, columns =colmn)
        print(one_hot_encoded_data.columns)
        one_hot_encoded_data=drop(one_hot_encoded_data)
        #one_hot_encoded_data = one_hot_encoded_data[:-212]
        one_hot_encoded_data=length_s*one_hot_encoded_data
        one_hot_encoded_data.to_numpy()
        X_t=np.c_[one_hot_encoded_data,length_s]
        return X_t
    size=test_Data.shape[0]
    train_Data=train_Data.append(test_Data)
    total=feature(train_Data)
    X_test=total[-size:]
    X_train=total[:-size]
    #X_train=feature(train_Data)
    #X_test=feature(test_Data)
    #X_test=X_test[:-212]
    def predict(X,Y,test_x,outputfile):
        X=np.c_[np.ones(X.shape[0]),X]
        trans=X.transpose()
        dot=np.dot(trans,X)
        inv=np.linalg.inv(dot)
        W=np.matrix(np.dot(inv,np.dot(trans,Y)))
        W=W.transpose()
        test_x=np.c_[np.ones(test_x.shape[0]),test_x]
        y_test=np.dot(test_x,W)
        y_test=y_test.flatten()
        saved_out = list(zip(y_test))
        df = pd.DataFrame(np.transpose(saved_out[0][0][0][0][0]))
        df.to_csv(outputfile,index = False, header=False)
        return outputfile
    return predict(X_train,Y,X_test,outputfile)


# In[10]:


if(len(sys.argv)==6 and sys.argv[1]=="a"):
    #trainfile.csv testfile.csv outputfile.txt weightfile.txt
    train_Data=pd.read_csv(sys.argv[2],index_col=[0])
    test_Data=pd.read_csv(sys.argv[3],index_col=[0])
    output_file=sys.argv[4]
    output_weight=sys.argv[5]
    LR_A(train_Data,test_Data,output_file,output_weight)
elif (len(sys.argv)==8 and sys.argv[1]=="b"):
    #trainfile.csv testfile.csv regularization.txt outputfile.txt weightfile.txt bestparameter.txt
    train_Data=pd.read_csv(sys.argv[2],index_col=[0])
    test_Data=pd.read_csv(sys.argv[3],index_col=[0])
    lambd=pd.read_csv(sys.argv[4],sep=",", header=None).to_numpy()
    output_file=sys.argv[5]
    output_weight=sys.argv[6]
    best_lambda=sys.argv[7]
    LR_B(train_Data,test_Data,lambd,output_file,output_weight,best_lambda)
elif(len(sys.argv)==5 and sys.argv[1]=="c"):
    #trainfile.csv testfile.csv outputfile.txt
    train_Data=pd.read_csv(sys.argv[2],index_col=[0])
    test_Data=pd.read_csv(sys.argv[3],index_col=[0])
    output_file=sys.argv[4]
    LR_C(train_Data,test_Data,output_file)
else:
    print("invalid",len(sys.argv))



# 
