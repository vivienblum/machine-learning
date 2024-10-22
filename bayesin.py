#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:03:32 2018

@author: vivien.blum
"""

import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def getTauxReussite(Y, res):
    imgRecognized = 0
    for index, result in enumerate(res):
        if result == Y[index]:
             imgRecognized += 1 
    return imgRecognized/Y.shape[0]

def getMatrix(X, Y):
    inv = []
    log = []
    rep = []
    i = 0
    while i < 10:
        cv = np.cov(X[Y == i], rowvar = 0)
        rep.append(X[Y == i].mean(axis = 0))
        
        log.append(np.linalg.slogdet(cv)[1])
        inv.append(np.linalg.inv(cv))
        
        i += 1
    return inv, log, rep

def getRes(X, inv, log, rep):
    res = []
    for index, img in enumerate(X):
        g = []
        
        for i in range(10):
            diff = (img - rep[i])
            g.append(-log[i] - np.dot(np.dot(np.transpose(diff), inv[i]), diff))
        res.append(np.argmax(np.array(g)))
    return res
    
CLASSIFIEURS = ["bayesien", "pca+gaussien", "vector", "neighbours", "pca+neighbours", "pca+vector"]

X0 = np.load('data/trn_img.npy')
Y0 = np.load('data/trn_lbl.npy')

X1 = np.load('data/dev_img.npy')
Y1 = np.load('data/dev_lbl.npy')

inv = []
log = []
rep = []

if "bayesien" in CLASSIFIEURS:
    print("** Bayésien **")
    timeTmp = time.time()
    print("[*] Apprentissage")
        
    inv, log, rep = getMatrix(X0, Y0)
    
    print("[*] Traitement")
    res = getRes(X1, inv, log, rep)
    
    print("[*] Calcul d'erreur")
    print("Taux réussite : " + str(getTauxReussite(Y1, res)*100) + "%")
    print("Temps : " + str(time.time() - timeTmp) + " sec")

if "pca+gaussien" in CLASSIFIEURS:
    print("** PCA + Gaussien **")
    timeTmp = time.time()
    print("[*] Apprentissage")
    
    myPca = PCA(n_components = 50).fit(X0)
    X0pca = myPca.transform(X0)
        
    inv, log, rep = getMatrix(X0pca, Y0)
    
    print("[*] Traitement")
    X1pca = myPca.transform(X1)
        
    res = getRes(X1pca, inv, log, rep)
        
    print("[*] Calcul d'erreur")
    print("Taux réussite : " + str(getTauxReussite(Y1, res)*100) + "%")
    print("Temps : " + str(time.time() - timeTmp) + " sec")

    print("[*] Enregistrement meilleur classifieur")
    np.save('test.npy', res)
    
    print("[*] Matrice de confusion")
    matConf = confusion_matrix(Y1, res)
    print(matConf)
    

if "vector" in CLASSIFIEURS:
    print("** Vector Machines (SVC) **")
    timeTmp = time.time()
    print("[*] Apprentissage")
    
    X0_Scale = preprocessing.scale(X0)
    clf = svm.SVC()
    clf.fit(X0_Scale, Y0)
    
    print("[*] Traitement")
    X1_Scale = preprocessing.scale(X1)
    res = clf.predict(X1_Scale)
    
    print("[*] Calcul d'erreur")
    print("Taux réussite : " + str(getTauxReussite(Y1, res)*100) + "%")
    print("Temps : " + str(time.time() - timeTmp) + " sec")

if "neighbours" in CLASSIFIEURS:
    print("** Neighbours **")
    
    timeTmp = time.time()
    print("[*] Apprentissage")
    
    neighbour = KNeighborsClassifier(n_neighbors = 3)
    neighbour.fit(X0, Y0)
    
    print("[*] Traitement")
    res = neighbour.predict(X1)
    
    print("[*] Calcul d'erreur")
    print("Taux réussite : " + str(getTauxReussite(Y1, res)*100) + "%")
    print("Temps : " + str(time.time() - timeTmp) + " sec")
    
if "pca+neighbours" in CLASSIFIEURS:
    print("** PCA + Neighbours **")
    
    timeTmp = time.time()
    print("[*] Apprentissage")
    myPca = PCA(n_components = 50).fit(X0)
    X0pca = myPca.transform(X0)
    
    neighbour = KNeighborsClassifier(n_neighbors = 3)
    neighbour.fit(X0pca, Y0)
    
    print("[*] Traitement")
    X1pca = myPca.transform(X1)
    res = neighbour.predict(X1pca)
    
    print("[*] Calcul d'erreur")
    print("Taux réussite : " + str(getTauxReussite(Y1, res)*100) + "%")
    print("Temps : " + str(time.time() - timeTmp) + " sec")
    
if "pca+vector" in CLASSIFIEURS:
    print("** PCA + SVC **")
    
    timeTmp = time.time()
    print("[*] Apprentissage")
    
    myPca = PCA(n_components = 50).fit(X0)
    X0pca = myPca.transform(X0)
    X0scale = preprocessing.scale(X0pca)
    
    clf = svm.SVC()
    clf.fit(X0scale, Y0)
    
    neighbour = KNeighborsClassifier(n_neighbors = 3)
    neighbour.fit(X0scale, Y0)
    
    print("[*] Traitement")
    X1pca = myPca.transform(X1)
    X1scale = preprocessing.scale(X1pca)
    
    res = neighbour.predict(X1scale)
    
    print("[*] Calcul d'erreur")
    print("Taux réussite : " + str(getTauxReussite(Y1, res)*100) + "%")
    print("Temps : " + str(time.time() - timeTmp) + " sec")