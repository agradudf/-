import numpy as np
import pandas as pd
import os

os.chdir(r'D:\Route 345')
recordings=['baseline_1','baseline_2','damage_1','damage_2','damage_3','damage_4','damage_5','damage_6']
test_group=['test_1','test_2','test_3']

def max_minnormalize(dataset,max_value,min_value):
    dataset=dataset.reshape(-1,60)

    dataset=2*(dataset-min_value)/(max_value-min_value)-np.ones(min_value.shape)

    return np.array(dataset.reshape(-1,256,60))

def sliding_window(x_data,X,sw_width=256,in_start=0):
    global s
    for _ in range(len(x_data)):
        in_end=in_start+sw_width
        if in_end<len(x_data):
            X.append(x_data[in_start:in_end])
        in_start+=32
    max_l=np.max(np.array(X,dtype='float32'),axis=(0,1))
    min_l=np.min(np.array(X,dtype='float32'),axis=(0,1))
    return X,max_l,min_l
X=[]
for i in range(0,8):
    for j in range(3):
        if 'baseline_1' in recordings[i]:
            df=pd.read_excel(recordings[i]+'_'+test_group[j]+'.xlsx')
            ar_x=df.values[:,1:61]
            X,max_value,min_value=sliding_window(ar_x,X,sw_width=256,in_start=0)
            print(max_value,min_value)
        
data_x0=np.array(X,dtype='float32')
x0=max_minnormalize(data_x0,max_value,min_value)
np.save('b1_DATA',x0)

X=[]
for i in range(0,8):
    for j in range(3):
        if 'baseline_2' in recordings[i]:
            df=pd.read_excel(recordings[i]+'_'+test_group[j]+'.xlsx')
            ar_x=df.values[:,1:61]
            X,_,_=sliding_window(ar_x,X,sw_width=256,in_start=0)

data_x1=np.array(X,dtype='float32')
x1=max_minnormalize(data_x1,max_value,min_value)
np.save('b2_DATA',x1)

X=[]
for i in range(0,8):
    for j in range(3):
        if 'damage_1' in recordings[i]:
            df=pd.read_excel(recordings[i]+'_'+test_group[j]+'.xlsx')
            ar_x=df.values[:,1:61]
            X,_,_=sliding_window(ar_x,X,sw_width=256,in_start=0)

data_x2=np.array(X,dtype='float32')
x2=max_minnormalize(data_x2,max_value,min_value)
np.save('d1_DATA',x2)

X=[]
for i in range(0,8):
    for j in range(3):
        if 'damage_2' in recordings[i]:
            df=pd.read_excel(recordings[i]+'_'+test_group[j]+'.xlsx')
            ar_x=df.values[:,1:61]
            X,_,_=sliding_window(ar_x,X,sw_width=256,in_start=0)

data_x3=np.array(X,dtype='float32')
x3=max_minnormalize(data_x3,max_value,min_value)
np.save('d2_DATA',x3)

X=[]
for i in range(0,8):
    for j in range(3):
        if 'damage_3' in recordings[i]:
            df=pd.read_excel(recordings[i]+'_'+test_group[j]+'.xlsx')
            ar_x=df.values[:,1:61]
            X,_,_=sliding_window(ar_x,X,sw_width=256,in_start=0)

data_x4=np.array(X,dtype='float32')
x4=max_minnormalize(data_x4,max_value,min_value)
np.save('d3_DATA',x4)

X=[]
for i in range(0,8):
    for j in range(3):
        if 'damage_4' in recordings[i]:
            df=pd.read_excel(recordings[i]+'_'+test_group[j]+'.xlsx')
            ar_x=df.values[:,1:61]
            X,_,_=sliding_window(ar_x,X,sw_width=256,in_start=0)

data_x5=np.array(X,dtype='float32')
x5=max_minnormalize(data_x5,max_value,min_value)
np.save('d4_DATA',x5)

X=[]
for i in range(0,8):
    for j in range(3):
        if 'damage_5' in recordings[i]:
            df=pd.read_excel(recordings[i]+'_'+test_group[j]+'.xlsx')
            ar_x=df.values[:,1:61]
            X,_,_=sliding_window(ar_x,X,sw_width=256,in_start=0)

data_x6=np.array(X,dtype='float32')
x6=max_minnormalize(data_x6,max_value,min_value)
np.save('d5_DATA',x6)

X=[]
for i in range(0,8):
    for j in range(2):
        if 'damage_6' in recordings[i]:
            df=pd.read_excel(recordings[i]+'_'+test_group[j]+'.xlsx')
            ar_x=df.values[:,1:61]
            X,_,_=sliding_window(ar_x,X,sw_width=256,in_start=0)

data_x7=np.array(X,dtype='float32')
x7=max_minnormalize(data_x7,max_value,min_value)
np.save('d6_DATA',x7)
