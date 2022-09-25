#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 12:31:52 2021

@author: luna
"""


import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LassoCV, Lasso, RidgeCV
from sklearn.metrics import adjusted_rand_score
import heapq
from sklearn.metrics import confusion_matrix, recall_score,precision_score,f1_score,matthews_corrcoef
import datetime
import random
from chainer import functions as F

#from _base import SGLBaseEstimator
from sgl import SGLCV


def criteria_bic(sse, beta_hat,n):
    beta_sig = np.sign(beta_hat**2).sum()
    bic = n*np.log(sse/n)+0.7*beta_sig*np.log(n)
    return bic


def k_bic(sse,k,sig_p,n,p):
    bic = np.log(sse/n)+np.log(np.log(p*k))*np.log(n)/n*sig_p
    return bic


def norm2(x):
    return np.sqrt((x**2).sum())
   
    
def relocate(group_update,k):
    group_true_cnt = np.arange(1,k+1)
    group_tmp_cnt =  np.unique(group_update)
    group_lack = list(set(group_true_cnt) - set(group_tmp_cnt))
    gcount = pd.value_counts(group_update)
    if group_lack:
        for i in range(len(group_lack)):
            gcount = gcount.append(pd.Series(0,index = [group_lack[i]]))
    u = np.where(gcount<5)[0]
        
    if len(u):  
        spa = np.array(gcount.index[u])  
        nspa =  np.array(gcount.index[np.where(gcount > 10)[0]])  
        nspa_ind = np.where(np.in1d(group_update,list(nspa)))[0]
        for ispa in range(len(spa)):               
            relocate_ind = np.random.randint(0,len(nspa_ind),size = 5)
            group_update[nspa_ind[relocate_ind]] = spa[ispa]
            nspa_ind = np.delete(nspa_ind,relocate_ind)
    return group_update



def SparseLasso(X,Y,k=5):    
    lassoModel = LassoCV(cv=k).fit(X,Y)
    indi = np.where(lassoModel.alphas_ == lassoModel.alpha_)[0]
    maxcv = lassoModel.mse_path_[indi,:].mean() + lassoModel.mse_path_[indi,:].std()/np.sqrt(k)
    best = np.where(lassoModel.mse_path_.mean(axis=1) < maxcv)[0][0]
    lassoModel = Lasso(alpha = lassoModel.alphas_[best]).fit(X,Y)
    betaFit = lassoModel.coef_
    return betaFit



def rpe_huber_relative(group_rst,beta_rst,intercept_rst,X,Y):
    k = np.max(group_rst)
    n = len(Y)
    resi2 = np.zeros(n)
    for l in range(k):
        X_tmp = X[:,group_rst == l+1]
        Y_tmp = Y[group_rst == l+1]
        resi2[group_rst==l+1] = abs(F.huber_loss(beta_rst[:,l].dot(X_tmp), Y_tmp-intercept_rst[l], delta=1.0,reduce='no').data/Y_tmp)
        #resi2[group_p1==group_set[l]] = (Y_tmp - beta_final[:,l].dot(X_tmp))**2
    return np.mean(resi2)


def rpe_relative(group_rst,beta_rst,intercept_rst,X,Y):
    k = np.max(group_rst)
    n = len(Y)
    resi2 = np.zeros(n)
    for l in range(k):
        X_tmp = X[:,group_rst == l+1]
        Y_tmp = Y[group_rst == l+1]
        resi2[group_rst==l+1] = abs((Y_tmp - beta_rst[:,l].dot(X_tmp)-intercept_rst[l])/Y_tmp)
    return np.mean(resi2)



def resi_calculate_hard(beta0, intercept_final, group1,X,Y):
    n = X.shape[1]
    group_set = np.unique(group1)
    resi2 = np.zeros(n)
    for l in range(len(group_set)):
        X_tmp = X[:,group1 == group_set[l]]
        Y_tmp = Y[group1 == group_set[l]]
        #resi2[group1 == group_set[l]] = (Y_tmp - beta0[:,l].dot(X_tmp))**2  
        resi2[group1 == group_set[l]] = F.huber_loss(beta0[:,l].dot(X_tmp), Y_tmp-intercept_final[l], delta=1.0,reduce='no').data
    resi_update = resi2.sum()
    return resi_update


def rmse_multi_hard(beta, beta_hat, group_hat, n,p,prop):    
    beta_kk = np.ones((n,p))
    beta_true = np.ones((int(n*prop[0]),p))*beta[:,0]
    for kk in range(1,beta.shape[1]):
        beta_true = np.vstack((beta_true,np.ones((int(n*prop[kk]),p))*beta[:,kk]))
    for kk in range(0,len(np.unique(group_hat))):
        beta_kk[group_hat == kk+1,:] = beta_hat[:,kk]
    rmse = np.sqrt(((beta_true-beta_kk)**2).sum()/n/p)
    
    return rmse


def confu_all(beta, beta_hat):
    s = np.hstack((np.sign(beta[:,0]**2),np.sign(beta[:,1]**2)))
    ps = np.hstack((np.sign(beta_hat[:,0]**2),np.sign(beta_hat[:,1]**2)))
    c1 = confusion_matrix(s,ps)

    s2 = np.hstack((np.sign(beta[:,1]**2),np.sign(beta[:,0]**2)))
    c2 = confusion_matrix(s2,ps)

    tp1 = c1[1,1]
    tp2 = c2[1,1]

    if tp1 > tp2:
        tp = tp1
        fp = c1[0,1]
        tpr = tp/sum(s)
        fpr = fp/sum(s==0)
        recall = recall_score(s,ps)
        precision = precision_score(s,ps)
        mcc = matthews_corrcoef(s,ps)
        fscore = f1_score(s,ps)
    else:
        tp = tp2
        fp = c2[0,1]
        tpr = tp/sum(s2)
        fpr = fp/sum(s2==0)
        recall = recall_score(s2,ps)
        precision = precision_score(s2,ps)
        mcc = matthews_corrcoef(s2,ps)
        fscore = f1_score(s2,ps)
    return tp, fp, tpr, fpr, recall, precision, mcc, fscore

  

def kmeans_ori_grp(X,Y,k,group_structure,n_groupsize,group_init = None):
    features, points = X.shape
    n_group = len(group_structure)

    dist = np.zeros((k, points))
    group_update = np.zeros(points)
    switched = True
    switched2 = True
    l_old = 1e5
    l_new = 1e4
    ite = 0
    if group_init is None:
        group_init = np.random.randint(1, k+1,size = points)
    
    group_init = relocate(group_init,k)
    group = group_init.copy()
    group_o2 = group.copy()
    
    #X_en,group_structure_en = enlarge_X(n_group,n_groupsize,X,group_structure)
    #features_en = X_en.shape[0]
    
    while switched2 and ite<50:
        group_set = np.unique(group)
        ite +=1  
        #print ("*******current iteration: " + str(ite)+ "**********")
        for i in range(len(group_set)):
            X_tmp = X[:,group == group_set[i]]
            Y_tmp = Y[group == group_set[i]]
            if ite <5:                
                # model = SGLCV(
                #     groups=group_structure_en, l1_ratio=[0.3,0.5,0.8], scale_l2_by='group_length', 
                #     alphas = [0.05], tol=1e-3, cv=3).fit(X_tmp.T, Y_tmp)
                model = SGLCV(
                    groups=group_structure, l1_ratio=[0.3,0.5,0.8], scale_l2_by='group_length', 
                    alphas = [0.05], tol=1e-3, eps=1e-2, cv=3,loss = 'squared_loss').fit(X_tmp.T, Y_tmp)

            else:
                # model = SGLCV(
                #     groups=group_structure_en, l1_ratio=[0.3,0.5,0.8], scale_l2_by='group_length', 
                #     n_alphas=40, tol=1e-3, eps=1e-2, cv=3).fit(X_tmp.T, Y_tmp)
                model = SGLCV(
                    groups=group_structure, l1_ratio=[0.3,0.5,0.8], scale_l2_by='group_length', 
                    n_alphas=20, tol=1e-2, eps=1e-2, cv=3,loss = 'squared_loss').fit(X_tmp.T, Y_tmp)

            beta_tmp = model.coef_
            beta_tmp[np.where(abs(beta_tmp)<0.1)[0]]=0
            intercept_tmp = model.intercept_

            dist[i,:] = (Y - beta_tmp.dot(X)-intercept_tmp)**2    #rows: group; columns: distance from center to point i 1 thru N
        
        l_old = l_new
        l_new = np.sum(np.min(dist,axis=0))

        for point in range(points):
            dt = dist[:,point].tolist()
            group_update[point] = dt.index(min(dt)) + 1
            group_update = group_update.astype(int)
        
        group_update = relocate(group_update,k)
        switched = sum(group_update != group)
        switched2 = sum(group_update != group)*sum(group_update != group_o2)
        group_o2 = group.copy()
        group = group_update.copy()


    group_set = np.unique(group_update)
    beta_final = np.zeros((features,len(group_set)))
    intercept_final = np.zeros(len(group_set))

    for i in range(len(group_set)):
        X_tmp = X[:,group_update == group_set[i]]
        Y_tmp = Y[group_update == group_set[i]]
        #beta_final[:,i] = SparseSGLasso3(X_tmp.T,Y_tmp,group_structure_en,loss='squared_loss')
        model = SGLCV(
                groups=group_structure, l1_ratio=[0.3,0.5,0.8], scale_l2_by='group_length', 
                n_alphas=40, tol=1e-3, eps=1e-2, cv=3,loss = 'squared_loss').fit(X_tmp.T, Y_tmp)
        beta_final[:,i] = model.coef_
        beta_final[np.where(abs(beta_final[:,i])<0.1)[0],i]=0       
        intercept_final[i] = model.intercept_

    # beta_sum = np.zeros((features,len(group_set)))
    # for i in range(len(group_set)): 
    #     beta_final_tmp = beta_final[:,i]
    #     beta_sum_tmp = np.zeros((features,n_group))
    #     for j in range(n_group):
    #         beta_sum_tmp[group_structure[j],j] = beta_final_tmp[group_structure[j]]
    #     beta_sum[:,i] = beta_sum_tmp.sum(axis=1)
   
    #print("k-means finished, iteration: " + str(ite))    
    return group_update, group_init, beta_final, intercept_final


def kmeans_robust_grp_base(X,Y,k,group_structure,n_groupsize,delta = 'adp',dista = 'huber',tt=50,group_init = None):
    features, points = X.shape
    n_group = len(group_structure)

    dist = np.zeros((k, points))
    l_new = 1e4
    l_old = 1e5
    group_update = np.zeros(points)
    switched = True
    ite = 0
    if group_init is None:
        group_init = np.random.randint(1, k+1,size = points)
    
    group_init = relocate(group_init,k)
    group = group_init.copy()
    
    #X_en,group_structure_en = enlarge_X(n_group,n_groupsize,X,group_structure)
    #features_en = X_en.shape[0]
    
    while abs(l_old-l_new)>1e-1 or ite<10:
        group_set = np.unique(group)
        ite +=1  
        for i in range(len(group_set)):
            X_tmp = X[:,group == group_set[i]]
            Y_tmp = Y[group == group_set[i]]
            if ite <15:                
                # model = SGLCV(
                #     groups=group_structure_en, l1_ratio=[0.3,0.5,0.8], scale_l2_by='group_length', 
                #     alphas = [0.05], tol=1e-3, cv=3).fit(X_tmp.T, Y_tmp)
                model = SGLCV(
                    groups=group_structure, l1_ratio=[0.3,0.5,0.8], scale_l2_by='group_length', 
                    alphas = [0.05], tol=1e-3, eps=1e-2, cv=3,loss = 'huber',delta = delta).fit(X_tmp.T, Y_tmp)

            else:
                model = SGLCV(
                    groups=group_structure, l1_ratio=[0.3,0.5,0.8], scale_l2_by='group_length', 
                    n_alphas=20, tol=1e-2, eps=1e-2, cv=3,loss = 'huber',delta = delta).fit(X_tmp.T, Y_tmp)
                
            beta_tmp = model.coef_
            beta_tmp[np.where(abs(beta_tmp)<0.1)[0]]=0
            intercept_tmp = model.intercept_
            
            if dista == 'huber':
                dist[i,:] = F.huber_loss(beta_tmp.dot(X), Y-intercept_tmp, delta=1.0,reduce='no').data
            if dista == 'sqr':
                dist[i,:] = (Y - beta_tmp.dot(X)-intercept_tmp)**2    #rows: group; columns: distance from center to point i 1 thru N
        
        l_old = l_new
        l_new = np.sum(np.min(dist,axis=0)) 


        for point in range(points):
            dt = dist[:,point].tolist()
            group_update[point] = dt.index(min(dt)) + 1
            group_update = group_update.astype(int)
        
        group_update = relocate(group_update,k)
        switched = sum(group_update != group)
        group = group_update.copy()

        if ite > tt:
            break

    group_set = np.unique(group_update)
    beta_final = np.zeros((features,len(group_set)))
    intercept_final = np.zeros(len(group_set))
    for i in range(len(group_set)):
        X_tmp = X[:,group_update == group_set[i]]
        Y_tmp = Y[group_update == group_set[i]]
        #beta_final[:,i] = SparseSGLasso(X_tmp.T,Y_tmp,group_structure_en,loss='huber')
        
        model = SGLCV(
                groups=group_structure, l1_ratio=[0.3,0.5,0.8], scale_l2_by='group_length', 
                n_alphas=40, tol=1e-3, eps=1e-2, cv=3,loss = 'huber',delta = delta).fit(X_tmp.T, Y_tmp)
            
        beta_final[:,i] = model.coef_
        beta_final[np.where(abs(beta_final[:,i])<0.1)[0],i]=0
        intercept_final[i] = model.intercept_


    return group_update, group_init, beta_final, intercept_final



def rep_kmeans_ori_grp(X, Y, k, group_true, beta, prop, group_structure,n_groupsize,rep_time = 10):
    bic_bst = 10000
    n = X.shape[1]
    p = X.shape[0]
    tresi = []
    tbic = []
    tari = []
    
    tfp=[]
    ttp=[]
    trmse=[]
    trecall = []
    tprecision = []
    tmcc = []
    tfscore = []
    

    group_rep = []
    starttime = datetime.datetime.now()
    for t in range(rep_time):
        group_p1,group_init, beta_final, intercept_final = kmeans_ori_grp(X,Y,k,group_structure,n_groupsize)
        ari = adjusted_rand_score(group_true, group_p1) 
        group_set = np.unique(group_p1)
        resi2 = np.zeros(n)
        for l in range(len(group_set)):
            X_tmp = X[:,group_p1 == group_set[l]]
            Y_tmp = Y[group_p1 == group_set[l]]
            resi2[group_p1==group_set[l]] = (Y_tmp - beta_final[:,l].dot(X_tmp) - intercept_final[l])**2           
        resi_update = resi2.sum()
        bic_update = criteria_bic(resi_update, beta_final,n)
        
        rmse_update = rmse_multi_hard(beta, beta_final, group_p1,n,p,prop)        
        tp_update, fp_update, tpr_update, fpr_update, recall_update, precision_update, mcc_update, fscore_update = confu_all(beta, beta_final)
        
        group_rep.append(group_p1)
        tari.append(ari)
        tresi.append(resi_update)
        tbic.append(bic_update)
        
        trmse.append(rmse_update)
        tfp.append(fpr_update)
        ttp.append(tpr_update)
        trecall.append(recall_update)
        tprecision.append(precision_update)
        tmcc.append(mcc_update)
        tfscore.append(fscore_update)


    #    if resi_update < resi_bst:
        if bic_update < bic_bst:
            group_bst = group_p1.copy()
            beta_bst = beta_final.copy()
            ari_bst = ari
            rpe_bst = np.sqrt(resi_update/n)
            bic_bst = bic_update
            intercept_bst = intercept_final
           
            rmse_bst = rmse_multi_hard(beta, beta_bst, group_bst,n,p,prop)
            tp_bst, fp_bst, tpr_bst, fpr_bst, recall_bst, precision_bst, mcc_bst, fscore_bst = confu_all(beta, beta_final)
        
    endtime = datetime.datetime.now()
    time = (endtime - starttime).total_seconds()    
    evalu = pd.DataFrame(np.vstack((ari_bst,rpe_bst,rmse_bst,tpr_bst,fpr_bst,recall_bst, precision_bst, mcc_bst, fscore_bst,bic_bst,time)).T,columns = ['ari','rpe','rmse','tpr','fpr','recall', 'precision', 'mcc', 'fscore','bic','time'])
    ttt = pd.DataFrame(np.vstack((tari,tresi,trmse,ttp,tfp,trecall,tprecision,tmcc,tfscore,tbic)).T,columns = ['ari','resi','rmse','tpr','fpr','recall', 'precision', 'mcc', 'fscore','bic'])

    return group_bst, beta_bst, intercept_bst, evalu, ttt, group_rep





def rep_kmeans_robust_grp(X, Y, k, group_true, beta, prop, group_structure,n_groupsize,delta = 'adp',dista = 'huber',rep_time = 10):
    bic_bst = 10000
    n = X.shape[1]
    p = X.shape[0]
    tresi = []
    tbic = []
    tari = []
    
    tfp=[]
    ttp=[]
    trmse=[]
    trecall = []
    tprecision = []
    tmcc = []
    tfscore = []

    group_rep = []
    starttime = datetime.datetime.now()
    for t in range(rep_time):
        group_p1, group_init, beta_final, intercept_final = kmeans_robust_grp_base(X, Y, k, group_structure=group_structure,n_groupsize = n_groupsize,delta = delta, dista = dista, tt=50)
        beta_final[abs(beta_final)<0.1] = 0
        ari = adjusted_rand_score(group_true, group_p1) 
        group_set = np.unique(group_p1)
        resi2 = np.zeros(n)
        for l in range(len(group_set)):
            X_tmp = X[:,group_p1 == group_set[l]]
            Y_tmp = Y[group_p1 == group_set[l]]
            resi2[group_p1==group_set[l]] = F.huber_loss(beta_final[:,l].dot(X_tmp), Y_tmp-intercept_final[l], delta=1.0,reduce='no').data
            #resi2[group_p1==group_set[l]] = (Y_tmp - beta_final[:,l].dot(X_tmp))**2           
        resi_update = resi2.sum()
        bic_update = criteria_bic(resi_update, beta_final,n)
       
        rmse_update = rmse_multi_hard(beta, beta_final, group_p1,n,p,prop)        
        tp_update, fp_update, tpr_update, fpr_update, recall_update, precision_update, mcc_update, fscore_update = confu_all(beta, beta_final)

        
        group_rep.append(group_p1)
        tari.append(ari)
        tresi.append(resi_update)
        tbic.append(bic_update)
        
        trmse.append(rmse_update)
        tfp.append(fpr_update)
        ttp.append(tpr_update)
        trecall.append(recall_update)
        tprecision.append(precision_update)
        tmcc.append(mcc_update)
        tfscore.append(fscore_update)


    #    if resi_update < resi_bst:
        if bic_update < bic_bst:
            group_bst = group_p1.copy()
            beta_bst = beta_final.copy()
            ari_bst = ari
            rpe_bst = np.sqrt(resi_update/n)
            bic_bst = bic_update
            intercept_bst = intercept_final

           
            rmse_bst = rmse_multi_hard(beta, beta_bst, group_bst,n,p,prop)
            tp_bst, fp_bst, tpr_bst, fpr_bst, recall_bst, precision_bst, mcc_bst, fscore_bst = confu_all(beta, beta_final)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).total_seconds()
    evalu = pd.DataFrame(np.vstack((ari_bst,rpe_bst,rmse_bst,tpr_bst,fpr_bst,recall_bst, precision_bst, mcc_bst, fscore_bst,bic_bst,time)).T,columns = ['ari','rpe','rmse','tpr','fpr','recall', 'precision', 'mcc', 'fscore','bic','time'])
    ttt = pd.DataFrame(np.vstack((tari,tresi,trmse,ttp,tfp,trecall,tprecision,tmcc,tfscore,tbic)).T,columns = ['ari','resi','rmse','tpr','fpr','recall', 'precision', 'mcc', 'fscore','bic'])

    return group_bst, beta_bst, intercept_bst, evalu, ttt, group_rep


