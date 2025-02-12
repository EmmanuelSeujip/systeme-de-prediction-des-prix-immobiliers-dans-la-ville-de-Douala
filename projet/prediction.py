import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

def prediction(quartier, type, annee):
    df=pd.read_excel('rassamble.xlsx',sheet_name=quartier)

    #suprression des colones inutiles
    df=df.drop(columns='A')


    #Transformation des donnée prise en donnée excel
    data=df.to_numpy()

    #On récupère les prix des appartements a 2 chambres
    prix=data[type]

    #supprime les prix
    data=np.delete(data, (5,6,7,8),axis=0)

    #convertir le tableau en un tableau de reels afin d'avoir des donnée plus precise 
    data=data.astype(np.float64)

    #normalisation du tableau grace a un parcourt de ligne et la methode z-score de la bibliothèque scipy
    for x in range(0,data.shape[0],1):
        data[x]=zscore(data[x])

    #creation du tableau pour le stokage des données

    #creation du tableau pour sauvegarder les différents II de chaque année
    saveTab=[]

    #On multiplie directement par 0.1 vu que c'est la même pondération
    dataPondere=data*0.1

    # creation du tableau du taux d'inflation pour chaque année
    inflation=[2.5,2.7,2.9,3.1,3.3,3,2.8,2.6,2.4,2.2,2.8,2.7,2.6,2.5,2.4,2.9,2.3,2.2,2.1,2.0,2.3,2.4,3.3,3.6,4.0]

    #remplissage des II
    for x in range(0,25,1):
        saveTab.append(np.sum(dataPondere[:,x])+(inflation[x]*0.5))


    # creation du tableau du prix par rapport a l'II 
    dataPrix=np.array((saveTab,prix))

    #creation du tableau de l'II par rapport a l'année
    dataAnnee=np.array((saveTab,[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]))



    #X1: variable des II, Y1: variable des prix
    X1=np.array(dataPrix[0])
    X1 = X1.reshape(-1, 1) #Pour eviter des erreurs sur la dimension du tableau
    Y1=np.array(dataPrix[1])

    #X2: variable des années, Y2: variable des II
    X2=np.array(dataAnnee[1])
    X2 = X2.reshape(-1, 1) #Pour eviter des erreurs sur la dimension du tableau
    Y2=np.array(dataAnnee[0])

    # Création des modèles de régression linéaire
    model1 = LinearRegression()
    model2 = LinearRegression()

    # Entraîner les modèles
    model1.fit(X1, Y1)
    model2.fit(X2, Y2)

    #Calcul du prix pour l'année 2030
    ii_annee= (model2.coef_[0])*annee + model2.intercept_ #c'est a dire Y2=aX2+b
    prix_annee=(model1.coef_[0])*ii_annee + model1.intercept_  #c'est a dire Y1=aX1+b
    return prix_annee
