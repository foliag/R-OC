# HALOS
This is a `Python` implementation of the following paper:
Sun, Y., Luo, Z., & Fan, X. (2022). Robust structured heterogeneity analysis approach for high-dimensional data. Statistics in medicine, 41(17), 3229–3259. https://doi.org/10.1002/sim.9414

# Introduction
We developed a robust structured heterogeneity analysis approach to identify subgroups, select important genes as well as estimate their effects on the phenotype of interest. Possible data contamination is accommodated by employing the Huber loss function. A sparse overlapping group lasso penalty is imposed to conduct regularization estimation and gene identification, while taking into account the possibly overlapping cluster structure of genes. This approach takes an iterative strategy in the similar spirit of K-means clustering. For more details, please check our paper.

# Requirements
```
-----
chainer             7.8.0
joblib              1.1.0
natsort             8.0.0
numpy               1.20.3
pandas              1.3.3
scipy               1.7.1
session_info        1.0.0
sklearn             0.24.1
skopt               0.9.0
tqdm                4.61.2
-----
IPython             7.29.0
jupyter_client      6.1.12
jupyter_core        4.8.1
notebook            6.4.5
-----
Python 3.9.5 (default, May 18 2021, 12:31:01) [Clang 10.0.0 ]
macOS-10.16-x86_64-i386-64bit
-----
```

# Demo
* Generate toy example data
```
import sys
sys.path.append(r'/HALOS/sourceROC')   #change the pathway to the source code's location

import rkmeans_source as rk
import numpy as np
import pandas as pd
import random
import os
os.chdir("/HALOS")   #pathway to save the results

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
```

* Implement algorithm and save results

```
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
```
