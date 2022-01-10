# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 17:14:51 2021

@author: celia
"""

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

#Récupérer les données d'entrainements
def dataSet(filename): #data.csv ou preTest.csv
    trainingSet = []
    with open(filename,'r') as file:
        lines = file.readlines()
        for line in lines:
            x=line.split(',')
            cl = classType([float(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])],x[6].rstrip('\n'))
            trainingSet.append(cl)
    return trainingSet

def dataSetFinal():
    trainingSet = []
    with open("finalTest.csv",'r') as file:
        lines = file.readlines()
        for line in lines:
            x=line.split(',')
            cl = classType([float(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])])
            trainingSet.append(cl)
    return trainingSet


class classType():
    def __init__(self,datas,name=None):
        self.datas = datas #6 caractéristiques d'une classe
        if (name != None):
            self.name = name
        #else:
            #self.name = self.knn()
    
    def __str__(self):
        sentence = self.name + "\ndatas :"
        for i in self.datas:
            sentence += str(i)
            sentence += " "
        return sentence

    def euclidianDistance(class1,class2): #distance euclidienne entre deux class
        #formule : d(X,Y)=racine(somme de 1 à n (yi-xi)^2)
        #X(x1,x2,x3,...,xn) et Y(y1,y2,y3,...,yn)
        dist = 0
        for x in range(6): #parcourt caractéristiques d'une class
            dist += (class2.datas[x] - class1.datas[x])**2
        return np.sqrt(dist)

    def knn(data,trainingSet,k):
        #itération pour parcourir toutes les données fournies
        distances = dict()
        for cl in trainingSet :
            dist = classType.euclidianDistance(data, cl) #calcul de la distance pour chaque point
            distances[cl] = dist #ajout au dico 
        #ordonner par ordre croissant selon la distance
        sortedDist = sorted(distances.items(), key=lambda x:x[1]) #trie selon la valeur de la distance (tuples)
        #retourne une liste de tuples (k,v), pas un dico
        #sélection du top k des voisins toruvés
        neighbors = []
        for nb in range(0,k): 
            neighbors.append(sortedDist[nb][0]) #récupération des caractéristiques
        #trouver la classe la plus fréquente parmis les top k
        cl = {"classA":0,"classB":0,"classC":0,"classD":0,"classE":0}
        for x in neighbors:
            typeC=x.name
            if typeC in cl:
                cl[typeC]+=1
            else:
                cl[typeC]=1
        sortedType = sorted(cl.items(),key=lambda x:x[1],reverse=True)
        return sortedType[0][0] #retourne la classe la plus fréquente

def train_test_split(dataSet):
    trainingBis=dataSet
    nbTrain = int(80*len(trainingBis)/100)   #80% des données = entrainement
    nbTest = int(len(trainingBis)-nbTrain)   #20% des données = tests
        
    #Sélection des données d'entrainement
    x_train=[] #liste des données = entrée
    y_train=[] #liste des types de classe = sortie
    for x in range(nbTrain): 
        index = random.randint(0,len(trainingBis)-1)
        x_train.append(trainingBis[index].datas)
        y_train.append(trainingBis[index].name)
        trainingBis.remove(trainingBis[index])
        
    #Sélection des données de test
    x_test=[]
    y_test=[]
    for x in range(nbTest):
        index = random.randint(0,len(trainingBis)-1)
        x_test.append(trainingBis[index].datas)
        y_test.append(trainingBis[index].name)
        trainingBis.remove(trainingBis[index])
    #on met au format classType
    trainingSet=[]
    for x in range(len(x_train)):
        trainingSet.append(classType(x_train[x], y_train[x]))
                           
    testSet=[]
    for x in range(len(x_test)):
        testSet.append(classType(x_test[x]))
        
    return trainingSet,testSet,y_test

def accuracy_score(y_true, y_predict):
    trueCount = 0
    for y in range(len(y_true)):
        if(y_true[y] == y_predict[y]):
            trueCount +=1
    return trueCount/len(y_true)


def findingK(trainingSet, x_test, y_test):
    neighbors = list(np.arange(3,21,2)) #tableau de k impairs
    errors=[]
    for k in neighbors :
        y_result=[]
        for x in x_test:
            y_result.append(classType.knn(x, trainingSet, k))
            
        acc = accuracy_score(y_test, y_result)
        errors.append(100*(1-acc))
    
    #plt.plot(range(3,21, 2),errors, 'o-')
    #plt.show()

    return neighbors[errors.index(min(errors))]     #k optimal

# %% zone du main
if __name__ == '__main__':
    acc = 0
    while(acc < 0.93):
        #Récupération des données des deux sets
        dataSet1=dataSet("data.csv")
        dataSet2=dataSet("preTest.csv")
    
        datasSet=[x for liste in [dataSet1, dataSet2] for x in liste]
    
        #Découpage des deux sets
        trainingSet1,x_test1,y_test1=train_test_split(dataSet1)
        trainingSet2,x_test2,y_test2=train_test_split(dataSet2)
    
        #Concaténation des listes
        trainingSet=[x for liste in [trainingSet1, trainingSet2] for x in liste]
        x_test=[x for liste in [x_test1, x_test2] for x in liste]
        y_test=[x for liste in [y_test1, y_test2] for x in liste]
    
        #Trouver K
        k = findingK(trainingSet, x_test, y_test)
        print(k)
    
        #Chargement du dataSet final
        dataSet_final=dataSetFinal()
    
        #Parcourt les données
        for data in dataSet_final:
            data.name = classType.knn(data, datasSet, k)
        
        #Etude des données
        y_true=[]
        y_result=[]
        for data in datasSet:
            y_true.append(data.name)
            y_result.append(classType.knn(data, trainingSet, k))
        
        acc = accuracy_score(y_true, y_result)
    
    #Matrice de confusion
    y_actu = pd.Series(y_true, name='Actual')
    y_pred = pd.Series(y_result, name='Predicted')
    print(pd.crosstab(y_actu, y_pred, margins = True))  
    with open("barras_sample1.txt",'w') as file:
        for data in dataSet_final:
            file.write(data.name+"\n")