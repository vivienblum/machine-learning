#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:03:32 2018

@author: vivien.blum
"""

import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ENV = "trn"

X0 = np.load('data/'+ENV+'_img.npy')
Y0 = np.load('data/'+ENV+'_lbl.npy')

#app = []
#avg = []
#cov = []
inv = []
log = []
rep = []

print("** Bayésien **")
time1 = time.time()
print("[*] Apprentissage")
i = 0
while i < 10:
    #app.append(X0[Y0 == i])
    #avg.append(app[i].mean(axis = 0))
    
    #cov.append(np.cov(app[i], rowvar = 0))
    cv = np.cov(X0[Y0 == i], rowvar = 0)
    rep.append(X0[Y0 == i].mean(axis = 0))
    
    
    log.append(np.linalg.slogdet(cv)[1])
    inv.append(np.linalg.inv(cv))
    
    i += 1
    
#On charge le dev
X1 = np.load('data/dev_img.npy')
Y1 = np.load('data/dev_lbl.npy')

print("[*] Traitement")
res = []
for index, img in enumerate(X1):
    g = []
    
    for i in range(10):
        diff = (img - rep[i])
        g.append(-log[i] - np.dot(np.dot(np.transpose(diff), inv[i]), diff))
    res.append(np.argmax(np.array(g)))

print("[*] Calcul d'erreur")
imgRecognized = 0
for index, result in enumerate(res):
    if result == Y1[index]:
         imgRecognized += 1       
        
taux = imgRecognized/X1.shape[0]
print("Taux réussite : " + str(taux))
print("Temps : " + str(time.time() - time1))

print("** PCA + Gaussien **")
time2 = time.time()
print("[*] Apprentissage")

myPca = PCA(n_components = 50).fit(X0)
X0pca = myPca.transform(X0)

invPca = []
logPca = []
repPca = []

i = 0
while i < 10:
    cvPca = np.cov(X0pca[Y0 == i], rowvar = 0)
    repPca.append(X0pca[Y0 == i].mean(axis = 0))
    
    logPca.append(np.linalg.slogdet(cvPca)[1])
    invPca.append(np.linalg.inv(cvPca))
    
    i += 1
    

print("[*] Traitement")
X1pca = myPca.transform(X1)
res = []
for index, img in enumerate(X1pca):
    g = []
    
    for i in range(10):
        diff = (img - repPca[i])
        g.append(-logPca[i] - np.dot(np.dot(np.transpose(diff), invPca[i]), diff))
    res.append(np.argmax(np.array(g)))
    
print("[*] Calcul d'erreur")
imgRecognized = 0
for index, result in enumerate(res):
    if result == Y1[index]:
         imgRecognized += 1       
        
taux = imgRecognized/X1.shape[0]
print("Taux réussite : " + str(taux))
print("Temps : " + str(time.time() - time2))

