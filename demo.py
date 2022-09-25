#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 19:53:45 2022

@author: luna
"""

import sys
sys.path.append(r'/Users/luna/lunas/project/kmeans2.0/github/sourceROC')   #change the pathway to the source code's location


import rkmeans_source as rk
import numpy as np
import pandas as pd
import random
import os


os.chdir("/Users/luna/lunas/project/kmeans2.0/github")   #where the results store



def gen_beta(p,a):
    q = a.shape[1]        # significant coefficients
    beta = np.zeros((p,k))
    beta[0:q,:] = a.T
    return beta




def gen_var_mono(n,p,rho,prop,beta,s,error='N'):
    sigma = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            sigma[i][j]=rho**abs(i-j)
    nprop = np.round(prop*n).astype(int)
    ntmp = np.insert(np.cumsum(nprop),0,0)
    np.random.seed(s)
    X = np.random.multivariate_normal(np.array([0]*p),sigma,n).T
    Y = np.array(())   
    group = np.array(())
    err = np.array(())
    for i in range(k):
        if error == 'N':
            err_tmp = 0.5*np.random.randn(nprop[i])
        if error == 'T':
            err_tmp = 0.5*np.random.standard_t(1, size=nprop[i])
        Y = np.append(Y,X[:,ntmp[i]:ntmp[i+1]].T.dot(beta[:,i].T) + err_tmp)
        group = np.append(group, np.ones(nprop[i])*i+1)
        err = np.append(err, err_tmp)
    group = group.astype(int)
    return Y, X, group, err


def gen_var_mix(n,p,rho,prop,beta,s):
    sigma = np.zeros(p*p).reshape(p,p)
    for i in range(p):
        for j in range(p):
            sigma[i][j]=rho**abs(i-j)
    nprop = np.round(prop*n).astype(int)
    ntmp = np.insert(np.cumsum(nprop),0,0)
    np.random.seed(s)
    X = np.random.multivariate_normal(np.array([0]*p),sigma,n).T
    Y = np.array(())   
    group = np.array(())
    err = np.array(())
    for i in range(k):
        err_tmp1 = 0.5*np.random.randn(int(round(0.7*nprop[i])))
        err_tmp2 = 0.5*np.random.standard_t(1, size=int(0.3*nprop[i]))
        err_tmp = np.hstack((err_tmp1,err_tmp2))
        Y = np.append(Y,X[:,ntmp[i]:ntmp[i+1]].T.dot(beta[:,i].T) + err_tmp)
        group = np.append(group, np.ones(nprop[i])*i+1)
        err = np.append(err, err_tmp)
    group = group.astype(int)
    return Y, X, group, err



###################
##start from here##
###################



####a toy simulation
k=2
n_sample = 300
n_group0 = 10
n_groupsize0 =  np.repeat(5,n_group0)


n_overlap = 3
p = sum(n_groupsize0) - n_overlap*(n_group0-1)
rho = 0.5
prop = np.array((0.5,0.5))    #subgroups' sample size proportion

tmp = np.append(0,np.cumsum(n_groupsize0)[:-1])
group_structure0 = []
for i in range(len(tmp)):
    group_structure0.append(np.arange(tmp[i]-i*n_overlap,tmp[i]-i*n_overlap+n_groupsize0[i]))


p_sig_index1 = np.array((1,3,6,9,10,11))
p_sig_index2 = np.array((2,3,5,9,10,12))

coef1 = np.array([[2,1,0.5,-1,1.5,1]])
coef2 = np.array([[-2,1,-2,-1,0.5,1.5]])


p_sig_index_all = np.vstack((p_sig_index1,p_sig_index2))
p_sig = len(p_sig_index1)


sig_coef = np.vstack((coef1,coef2))

beta = np.zeros((p,k))
beta[p_sig_index1,0] = coef1
beta[p_sig_index2,1] = coef2



s = random.randint(0,100000000)+os.getpid()
Y,X,group_true, err = gen_var_mono(n_sample,p,rho,prop,beta,s,error='N')

print("current method: R-OC")
group_roc, beta_roc, intercept_roc, evalu_roc, select_roc, group_rep_roc = rk.rep_kmeans_robust_grp(X, Y, k, group_true, beta, prop, group_structure=group_structure0,n_groupsize = n_groupsize0,delta = 1.3, rep_time = 5)


print(evalu_roc)   #evaluation result
print(group_roc)   #subgroup identification result
beta_roc[:5,:]     #subgroup coefficients, column for each subgroup
print(select_roc)  #evaluation result for each initial value
print(group_rep_roc)   #subgroup identification result for each initial value


pd.DataFrame(beta_roc.T).to_csv("beta_roc.csv",mode = 'a',index =False,header = None)       
pd.DataFrame(group_roc.reshape(1,-1)).to_csv("group_roc.csv",mode = 'a',index =False,header = None)
evalu_roc.to_csv("evalu_roc.csv",mode = 'a',index =False,header = None)

####full simulation: (S6); Error ~ 𝑡(1)
k=2
n_sample = 500
n_group0 = 24
n_group1 = round(n_group0*0.5)
n_group2 = round(n_group0*0.2)
n_group3 = round(n_group0*0.3)

n_groupsize0 =  np.repeat(10,n_group0)


n_overlap1 = 2
n_overlap2 = 5
n_overlap3 = 0

df_list = [0.8]

p = sum(n_groupsize0) - n_overlap1*(n_group1-1)-n_overlap2*(n_group2-1)-n_overlap3*(n_group3-1) +1
rho = 0.5
prop = np.array((0.5,0.5))    #subgroups' sample size proportion

tmp = np.append(0,np.cumsum(n_groupsize0)[:-1])
group_structure0 = [np.arange(0,n_groupsize0[0])]
for i in range(len(tmp)):
    if (i-1)<n_group1:
        group_structure0.append(np.arange(group_structure0[i][-1]-n_overlap1+1,group_structure0[i][-1]-n_overlap1+1+n_groupsize0[i]))
    if n_group1<=(i-1)<(n_group2+n_group1):
        group_structure0.append(np.arange(group_structure0[i][-1]-n_overlap2+1,group_structure0[i][-1]-n_overlap2+1+n_groupsize0[i]))
    if (i-1)>=(n_group1+n_group2):
        group_structure0.append(np.arange(group_structure0[i][-1]-n_overlap3+1,group_structure0[i][-1]-n_overlap3+1+n_groupsize0[i]))



p_sig_index1 = np.array((1,3,6,9,10,11,13,16,20,21,25,27,29,31,33))
p_sig_index2 = np.array((2,3,5,9,10,12,14,17,18,19,22,23,26,31,33))

coef1 = np.array([[2,1,0.5,-1,1.5,0.5,-1,2,-1,0.5,-1,0.5,1.5,0.5,1]])
coef2 = np.array([[-2,1,-2,-1,0.5,1.5,1,-1,-0.5,2,-0.5,1,-2,-0.5,-1]])


p_sig_index_all = np.vstack((p_sig_index1,p_sig_index2))
p_sig = len(p_sig_index1)


sig_coef = np.vstack((coef1,coef2))

beta = np.zeros((p,k))
beta[p_sig_index1,0] = coef1
beta[p_sig_index2,1] = coef2
pd.DataFrame(beta).to_excel("beta_true.xlsx")



s = random.randint(0,100000000)+os.getpid()
Y,X,group_true, err = gen_var_mono(n_sample,p,rho,prop,beta,s,error='T')    
#Y,X,group_true, err = gen_var_mix(n_sample,p,rho,prop,beta,s)
pd.DataFrame(X.reshape(1,-1)).to_csv("X_all.csv",mode = 'a',index =False,header = None)
pd.DataFrame(Y.reshape(1,-1)).to_csv("Y_all.csv",mode = 'a',index =False,header = None)   

n_group_s = n_group0
n_groupsize_s =  n_groupsize0.copy()
group_structure_s = group_structure0.copy()


print("current method: R-OC")
group_roc, beta_roc, intercept_roc, evalu_roc, select_roc, group_rep_roc = rk.rep_kmeans_robust_grp(X, Y, k, group_true, beta, prop, group_structure=group_structure_s,n_groupsize = n_groupsize_s,delta = 1.3, rep_time = 20)


print(evalu_roc)   #evaluation result
print(group_roc)   #subgroup identification result
beta_roc[:5,:]     #subgroup coefficients, column for each subgroup
print(select_roc)  #evaluation result for each initial value
print(group_rep_roc)   #subgroup identification result for each initial value

pd.DataFrame(beta_roc.T).to_csv("beta_roc.csv",mode = 'a',index =False,header = None)       
pd.DataFrame(group_roc.reshape(1,-1)).to_csv("group_roc.csv",mode = 'a',index =False,header = None)
evalu_roc.to_csv("evalu_roc.csv",mode = 'a',index =False,header = None)



