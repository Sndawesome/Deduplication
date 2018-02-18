# import libraries
import numpy as np
import pandas as pd
import re, string

#Loading Dataset
train=pd.read_csv("Deduplication Problem - Sample Dataset.csv")
#Make copy of dataset
o=train.copy()
#Splitting ln and fn column 
train=pd.concat([train,train.ln.str.split(expand=True)],axis=1)
train=train.rename(columns={0:"ln_1",1:"ln_2"})
train=pd.concat([train,train.fn.str.split(expand=True)],axis=1)

# Building new feature of first letter of fn and ln 
def f(x):
    if x is not None:
        return x[0]
    else: return x
train[1.1]=list(map(f,train[1]))
train['ln_2_1']=list(map(f,train['ln_2']))
train[0.1]=list(map(f,train[0]))
train['ln_1_1']=list(map(f,train['ln_1']))
#Drop the fn and ln column
train.drop(['ln','fn'],axis=1,inplace=True)

#Remove duplicates based on first name,last name,dob,gn
x=train.drop_duplicates(['dob','gn','ln_1','ln_2',0,1])
#x=x.drop_duplicates(['dob','gn','ln_1','ln_2_1',0,1,1.1])
x.groupby(['dob','gn','ln_1',0]).count()

#Remove Duplicates based on name and last name suffix
z=x[x[['dob','gn','ln_1',0]].duplicated(keep=False)]
z['ln_2'].fillna(1,inplace=True)
z[1].fillna(0,inplace=True)
x.drop(list(z[(z[1]==0) & (z.ln_2==1)].index),inplace=True)
n=z[z[['dob','gn','ln_1','ln_2',0,1.1]].duplicated(keep=False)]
x.drop(list(n[n[1].apply(lambda x:len(x)==1)].index),inplace=True)
m=z[z[['dob','gn','ln_1',0,'ln_2_1',1]].duplicated(keep=False)]
x.drop(list(m[m.ln_2.apply(lambda x:len(x)==1)].index),inplace=True)


#REMOVE DUPLICATES BASED ON INITIALS OF FN AND LN
t=x[x[['dob','gn','ln_1_1',0]].duplicated(keep=False)]
x.drop(list(t[t.ln_1.apply(lambda x:len(x)==1)].index),inplace=True)
p=x[x[['dob','gn','ln_1',0.1]].duplicated(keep=False)]
x.drop(list(p[p[0].apply(lambda x:len(x)==1)].index),inplace=True)


#Final output
out=o.iloc[list(x.index)]
out.to_csv('Deduplication-Output.csv',index=0)


