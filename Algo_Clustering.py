
# idées :
# - demontrer la terminaison des clusterings
# graphe inertia selon nbr groupes
# mathematique autour de l'algo ( voir ite internet ) => meilleur ini


## Algo Cluster

from math import sqrt
from random import *

def centroides (dataset,n):
    A = []
    for k in range (n):
        a = choice(dataset)
        A.append(a)
    return A

def meilleurs_centroides (dataset): #  idées : meilleur centroides = points les plus éloignés possibles
                                    # à refaire car la pas ouf
    A = []
    t = []
    p = []
    for k in range (len(dataset)):
        m = []
        for i  in range(len(dataset)):
            l=0
            res = 0
            while l < len(dataset[0])-1:
                res += (dataset[k][l]-dataset[i][l])**2
                l = l+1
            m.append([sqrt(res)])
        t.append(m.index(max(m)))
        p.append(max(m))
    a = p.index(max(p))
    A.append (dataset[a])
    A.append (dataset[t[a]])
    return A

def cluster (A,dataset):
    t = []
    m = []
    for k in range (len(dataset)):
        t = []
        for i in range (len(A)):
            l=0
            res = 0
            while l < len(A[0]):
                res += (A[i][l]-dataset[k][l])**2
                l = l+1
            t.append(sqrt(res))
        m.append(t.index(min(t)))
    return m


def new_cluster (A,dataset):
    L = cluster(A,dataset)
    new_A = []
    m = [[] for i in range (len(A))]
    for k in range (len(m)):
        m[k] = [0 for i in range (len(A[0]))]
        m[k].append(0)
    for k in range (len(dataset)):
        a = L[k]
        b = dataset[k]
        for i in range (len(m[a])-1):
            m[a][i] += dataset[k][i]
        m[a][-1] += 1
    for k in range (len(m)):
        for l in range (len(m[0])):
            if m[k][-1] > 0 : m[k][l]= m[k][l]/m[k][-1]
    for k in range (len(m)):
        p = [m[k][i] for i in range (len(m[k])-1)]
        new_A.append(p)
    return new_A



def inertia (A,dataset):
    inertia = 0
    for k in range (len(dataset)):
        t = []
        for i in range (len(A)):
            l=0
            res = 0
            while l < len(A[0]):
                res += (A[i][l]-dataset[k][l])**2
                l = l+1
            t.append(sqrt(res))
        inertia += min(t)
    return inertia


def regroupement_n(dataset,n):
    A = centroides(dataset,n)
    t = []
    t.append(inertia(A,dataset))
    A = new_cluster(A,dataset)
    t.append(inertia(A,dataset))
    while t[-1] != t[-2]:
        A = new_cluster(A,dataset)
        t.append(inertia(A,dataset))
    clusters = cluster(A,dataset)
    return [clusters,A]

def regroupement_opti(dataset):
    A = meilleurs_centroides(dataset)
    t = []
    t.append(inertia(A,dataset))
    A = new_cluster(A,dataset)
    t.append(inertia(A,dataset))
    while t[-1] != t[-2]:
        A = new_cluster(A,dataset)
        t.append(inertia(A,dataset))
    clusters = cluster(A,dataset)
    return clusters



## Test

def datatest(dataset):
    datatest = []
    for k in range (len(dataset)):
        m = []
        for l in range (len(dataset[k])-1):
            m.append(dataset[k][l])
        datatest.append(m)
    return datatest

def traitement(L):
    m = []
    for k in range (len(L)):
        if L[k] == 1 :
            m.append('M')
        else : m.append('B')
    return m

def traitement2(L,dataset,n):
    t = []
    m = [0 for i in range (n)]
    p = [reponse_test(dataset)[k] for k in range (100)]
    for  k  in range (100):
    # On utilise la connaisance sur 100 données pour determiner le resultat de tout le dataset, rmq : l'algo n'utilise pas les reps des resultats connu, ils peuvent donc faire parti des erreurs.
        if p[k] == 'B' :
            m[L[k]] += 1
        else : m[L[k]] -= 1
    for k in range (len(m)):
        if m[k]<0 :
            m[k]='M'
        else : m[k]='B'
    for k in range (len(dataset)):
        a = m[L[k]]
        t.append(a)
    return [t,m]



def reponse_test (P):
  return [P[k][-1] for k in range(len(P))]


def resultat_n(dataset,n):
    a = reponse_test(dataset)
    b = traitement2(regroupement_n(datatest(dataset),n)[0],dataset,n)[0]
    l = 0
    c = 0
    while l < len(dataset) :
        if a[l] != b[l] :
            c += 1
        l = l + 1
    return ((c/len(dataset))*100)

def res (dataset,d,n):
    a = reponse_test(dataset)
    b = traitement2(d,dataset,n)[0]
    l = 0
    c = 0
    while l < len(dataset) :
        if a[l] != b[l] :
            c += 1
        l = l + 1
    return ((c/len(dataset))*100)

def opti_centroides (dataset):
    t = []
    for k in range (2,11):
        m = []
        for i in range (50):
            m.append(resultat_n(dataset,k))
        t.append(min(m))
    print(t)
    return (min(t),t.index(min(t))+1)

# On trouve que le programme est optimisé pour 4 centroides

def resultat_opti (dataset):
    a = reponse_test(dataset)
    b = traitement(regroupement_opti(datatest(dataset)))
    l = 0
    c = 0
    while l < len(dataset) :
        if a[l] != b[l] :
            c += 1
        l = l + 1
    return ((c/len(dataset))*100)

def resultat_centroide (dataset,n):
    t = []
    for k in range (100):
        t.append(resultat_n(dataset,n))
    return(min(t))

## Résultats

# taille dataset :      350
#
#   nbr de centroides   /     erreurs
#
#        2               18.571428571428573 %
#        3               10.571428571428571 %
#        4               10.285714285714285 %
#        5               10.571428571428571 %
#        6               10.285714285714285 %
#        7               10.285714285714285 %
#        8
#
#     On remarque qu'il n'est pas utile de dépasser 4 centroides
# Avec une amélioration du dataset et 4 centroides, on a des résultat à 5,19% d'erreur mais une moyenne a 8,88%,
# Sur 200 calculs, 40 ont donné des réponses en dessous de 6% d'erreur,


def resultat_f(dataset,k):
    return resultat_n(dataset,k)

def stat_resultat_f (dataset,n,p):
    ans = 0
    c = 0
    t = []
    for k in range (n):
        x = resultat_f(dataset,p)
        ans += x
        t.append(x)
    return ( ans/n ,min(t))


def liste_min_k (dataset,n,m):
    t = []
    for k in range (2,m) :
        x,y = stat_resultat_f(dataset,n,k)
        t.append(y)
    return t

liste_min_wis = [8.787346221441124, 8.122636203866433, 5.448154657293498, 4.92091388400703, 4.92091388400703, 5.0966608084358525, 4.21792618629174, 4.042179261862917,3.690685413005272]
# nbr centroides  2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10
liste_min_mass = [20.240963855421686, 19.879518072289155, 19.759036144578314, 19.879518072289155,19.638554216867472]

#liste_min_mass =


# Dans la mesure ou cet algorithme est basé sur le choix aléatoire des points initiaux, on ne peut obtenir le même résultat à chaque fois, on pourrait en revanche analyser pour quels points les résultats sont les meilleurs et ensuite developper un alorithme les sélectionnant, en prenant ici l'hypothèse qu'il existerait une proprieté commune quel que soit le dataset comme par exmple les points les plus éloignés les uns des autres....

# En répétant l'algorithme un nombre n de fois, on peut séléctionner la fois où le résultat est le plus proche de celui escompté. Mais dans les faits, on ne connais pas les réponses, ainsi la partie connues des données, c'est à dire ici les 100 premiers éléments de la liste peuvent servir de repères, comme dit précédemment, leurs résultats de ces points est calculé par l'algorithme et non pas transféré depuis les données connues, ils permettent donc d'indiquer le cas le plus précis, et une meilleure réponse. Bien sur on aurait une incertitude car l'utulité de cette méthode est de diminuer la taille de la base de données en faisant un algorithme hybride entre labélisé et non labélisé, il faut donc une baase de donnée permettant d'indiquer le cas le plus précis mais assez petite pour garder l'interet du'une telle méthode.

# On remarque aussi que la précision est de un peu plus de 94,8 % ce qui est un très bon résultat, avec 100 valeurs connues. ( on estime ici que l'erreur statistique sur les valeurs connues ( dont on utilise pas la réponse ici ) est la même que celle des données cherchées ) .


def sortie_algo(dataset,liste,i,n):
    L = dataset
    t = []
    min = 1.1*liste[i-2]
    a = reponse_test(L)
    for k in range (n):
        b = traitement2(regroupement_n(datatest(L),i)[0],L,i)[0]
        l = 0
        c = 0
        while l < len(L) :
            if a[l] != b[l] :
                c += 1
            l = l + 1
        x = ((c/len(L))*100)
        if x < min :
            min = x
            t = b
    return [min,t]





def count(n,L):
    a = [0 for i in range (n)]
    for k in range (len(L)):
        a[L[k]] += 1
    return a

def count_rep (dataset):
    a = [0,0]
    data = reponse_test(dataset)
    for k in range (len(data)):
        if data[k] == 'M' :
            a[1] += 1
        else : a [0] += 1
    return a


import matplotlib.pyplot as plt



def ROC (dataset,liste,i):
    L = dataset
    l = len(L)
    a = sortie_algo(dataset,liste,i,400)[1]
    b = reponse_test(L)
    x = []
    y = []
    precision = []
    rappel = []
    FN = 0
    TP = 0
    TN = 0
    FP = 0
    for k in range (l):
        if b[k]=='B' :
            if a[k]=='B' : TP += 1
            else : FN += 1
            y.append(TP)
            x.append(FP)
            if TP+FP == 0 : precision.append(0)
            else : precision.append(TP/(TP+FP))
            if TP+FN == 0 : rappel.append(0)
            else : rappel.append(TP/(TP+FN))
        else :
            if a[k]=='M' : TN += 1
            else : FP += 1
            x.append(FP)
            y.append(TP)
            if TP+FP == 0 : precision.append(0)
            else : precision.append(TP/(TP+FP))
            if TP+FN == 0 : rappel.append(0)
            else : rappel.append(TP/(TP+FN))
    for k in range (len(x)):
        x[k]=x[k]/(FP+TN)
    for k in range (len(y)):
        y[k]=y[k]/(FN+TP)
    for k in range (len(y)):
        y[k]=y[k]/y[-1]
    x.append(1)
    y.append(1)
    PIPP = TP/(TP+FP)   # positifs_in_predits_positifs ( précision )
    PPIP = TP/(TP+FN)   # predits_positifs_in_positifs ( rappel )
    MH = 2*(PIPP*PPIP)/(PIPP+PPIP)  #moyenne_harmonique
    return [PIPP,PPIP,MH,(precision,rappel),(x,y)]

# Taux faux positifs = FP/(FP+TN)   ( 1 - spécificité )
# Taux vrais positifs = TP/(TP+FN)  ( sensitivité )

# On obtient :
# PIPP = 0.9479166666666666
# PPIP = 0.9578947368421052
#  MH  = 0.9528795811518324

def graphe_ROC(dataset,liste,i):
    x,y = ROC(dataset,liste,i)[-1]
    plt.plot(x,y,c = 'red',label='ROC')
    plt.title('Courbe de ROC')
    plt.xlabel('taux faux positifs')
    plt.ylabel('taux vrais positifs')
    plt.show()

def comparaison_graphe_ROC(dataset,liste,i,j):
    x,y = ROC(dataset,liste,i)[-1]
    x2,y2 = ROC(dataset,liste,j)[-1]
    plt.plot(x,y,c = 'red')
    plt.plot(x2,y2,c = 'b')
    plt.title('comparaison Courbe de ROC')
    plt.xlabel('taux faux positifs')
    plt.ylabel('taux vrais positifs')
    plt.show()

def graphe_ROC_precision_rappel (dataset,liste,i):  # à refaire
    precision,rappel = ROC(dataset,liste,i)[-2]
    plt.plot(rappel,precision,c = 'red',label='ROC')
    plt.title('precison = f(rappel)')
    plt.xlabel('rappel')
    plt.ylabel('precision')
    plt.show()

def aire_ROC(dataset,liste,i,n):  # on utilise ici la méthode de monte carlo ( proba )
    x,y = ROC (dataset,liste,i)[-1]
    c = 0
    for k in range (n):
        x0 = random()
        y0 = random()
        k = 0
        xk = 0
        while k < len(x) and xk < x0 :
            k = k+1
            xk = x[k]
        if y0 > y[k] : c += 1
    return 1-c/n

# Pour n = 10^4 on obtient une aire sous la courbe de 0.9857
# Pour 10^6 on obtient une aire de 0.985055
# on a donc une aire sous la courbe de pres de 98,5%

##   amelioration dataset

def tri_param (dataset,l):
    t = []
    for k in range (len(dataset)):
        t.append([dataset[k][i] for i in l])
    return t

def moyenne_dataset(dataset):
    M = [0 for i in range (30)]
    for k in range (len(dataset)):
        for l in range (len(dataset[k])-1):
            M[l] += dataset[k][l]
    for k in range(len(M)):
        M[k] = M[k]/(len(dataset))
    return M

# def rapport_moyenne_dataset(dataset):
#     t = []
#     M = moyenne_dataset(dataset)
#     for i in range (len(dataset)):
#         m = []
#         for k in range (len(M)):
#             m.append(dataset[i][k]/M[k])
#         m.append(dataset[i][-1])
#         t.append(m)
#     return t
#
# def tri_valeur_rapport_dataset (dataset,p):  # k = 8 marche bien
#     t = []
#     r = rapport_moyenne_dataset(dataset)
#     for k in range (len(r)):
#         B = True
#         for i in range (len(r[k])-1):
#             if r[k][i] > p :
#                 B = False
#         if B == True :  t.append(r[k])
#     return t
#
# def sur1 (dataset,p):
#     t = tri_valeur_rapport_dataset(dataset,p)
#     M = []
#     for i in range (len(t[0])-1):
#         l = []
#         for k in range (len(t)):
#             l.append(t[k][i])
#         M.append(max(l))
#     for k in range (len(t)):
#         for i in range (len(t[0])-1):
#             t[k][i] = t[k][i]/M[i]
#     return t






def moyenne_dataset(dataset):
    M = [0 for i in range (len(dataset[0])-1)]
    for k in range (len(dataset)):
        for l in range (len(dataset[k])-1):
            M[l] += dataset[k][l]
    for k in range(len(M)):
        M[k] = M[k]/(len(dataset))
    return M

def ecart_type (dataset):
    ecart = []
    m = moyenne_dataset(dataset)
    for l in range (len(dataset[0])-1):
        e = 0
        for k in range (len(dataset)):
            a = (m[l]-dataset[k][l])
            e += a*a
        ecart.append(sqrt(e/(len(dataset))))
    return ecart

def standardization (dataset):
    data = []
    m = moyenne_dataset(dataset)
    e = ecart_type(dataset)
    for k in range (len(dataset)):
        d = []
        for l in range (len(dataset[0])-1) :
           d.append((dataset[k][l] - m[l])/e[l])
        d.append(dataset[k][-1])
        data.append(d)
    return data


## Base de donnée



import random
import csv

def lecture_fichier():
    with open ('dataset_convert.csv','r') as base_donnee :
        csv_reader = csv.reader(base_donnee)
        dataset = []
        for line in (base_donnee):
            dataset.append(line.split(","))
        for k in range (len(dataset)):
            for l in range (len(dataset[k])):
                dataset[k][l] = float(dataset[k][l])
        return dataset

def shuffle (liste):
    l = len(liste)
    liste_shuffle = []
    for k in range (l):
        n = (l-1-k)
        r = random.randint(0,n)
        liste_shuffle.append(liste[r])
        liste.remove(liste[r])
    return liste_shuffle

def mise_en_forme(dataset):
    data = [dataset[k][2:]+[dataset[k][1],dataset[k][0]]for k in range (len(dataset))]
    return data

def mise_en_forme_temp(dataset):
    data = [dataset[k][2:]+['B' if dataset[k][1]==0 else 'M']for k in range (len(dataset))]
    return data


def lecture_fichier2():
    with open ('mammographic_mass.csv','r') as base_donnee :
        csv_reader = csv.reader(base_donnee)
        dataset = []
        for line in (base_donnee):
            dataset.append(line.split(","))
        for k in range (1,len(dataset)):
            for l in range (len(dataset[k])):
                dataset[
                k][l] = float(dataset[k][l])
        return dataset[1:]

dataset_mammographic_mass = lecture_fichier2()

def mise_en_forme2(dataset):
    data = [ dataset[k][:-1] + [ 'B' if dataset[k][-1] == 0 else 'M' ] for k in range (len(dataset)) ]
    return data



Base_de_données_wisconsin_originale = mise_en_forme_temp(lecture_fichier())
Base_de_données_wisconsin = shuffle(Base_de_données_wisconsin_originale)
stand_base_wis = standardization(Base_de_données_wisconsin)
stand_donnée_wis = stand_base_wis [:217]
stand_test_wis =  stand_base_wis [217:]



base_de_données_mass_originale = mise_en_forme2(dataset_mammographic_mass)
base_de_données_mass = shuffle(base_de_données_mass_originale)
stand_base_mass = standardization(base_de_données_mass)

stand_donnée_mass = stand_base_mass[:400]
stand_test_mass = stand_base_mass[400:]

# grace à proportion on mesure que 217 donne le meilleur resultat
# 12 erreur soit  3.418803418803419% d'erreur une fois les valeurs abérrantes enlevées


## PCA scikit learn
import matplotlib
import colorsys
from sklearn.decomposition import PCA

couleurs =  [ 'g', 'c', 'm', 'y', 'k', 'burlywood' , 'lightcoral', 'violet', 'pink' , 'darksalmon' ]



def visualisation_groupe_PCA(dataset,k,liste):
    a = datatest(dataset)
    c = 0
    d = regroupement_n(datatest(dataset),k)[0]
    x = res(dataset,d,k)
    seuil = liste[k-2]
    while x > seuil and c < 400:
        d = regroupement_n(datatest(dataset),k)[0]
        x = res(dataset,d,k)
        c += 1
    e = reponse_test(dataset)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(a)
    Y = principalComponents.tolist()
    fig1 = plt.figure(1)
    ax1 = fig1.gca()
    fig2 = plt.figure(2)
    ax2 = fig2.gca()
    for i in range (len(Y)) :
        if e[i] == 'M' :
            ax2.scatter(Y[i][0],Y[i][1],c='r')
        else : ax2.scatter(Y[i][0],Y[i][1],c='b')
    for k in range (len(Y)):
        ax1.scatter(Y[k][0],Y[k][1],c=couleurs[d[k]])
    plt.show()
    return x


## PCA fait maison


def mean (dataset):
    l = len(dataset)
    p = len(dataset[0])
    mean = [0 for i in range (p)]
    for k in range (l):
        for i in range (p):
            mean[i] += dataset[k][i]
    for i in range (p):
        mean[i] = mean[i]/l
    return mean

def variance (dataset,i,mean):
    l = len(dataset)
    res = 0
    mean_i = mean[i]
    for k in range(l):
        res += (dataset[k][i]-mean_i)*(dataset[k][i]-mean_i)
    return res/l

def covariance (dataset,i,j,liste_mean):
    l = len(dataset)
    res = 0
    mean_i = liste_mean[i]
    mean_j = liste_mean[j]
    for k in range(l):
        res += (dataset[k][i]-mean_i)*(dataset[k][j]-mean_j)
    return res/l

def matrice_covariance (dataset):
    liste_mean = mean(dataset)
    p = len(liste_mean)
    matrice_covariance = []
    for k in range (p):
        m = []
        for l in range (k+1):
            m.append(covariance(dataset,k,l,liste_mean))
        comble = [0 for i in range (p-k-1)]
        matrice_covariance.append(m+comble)
    for k in range (p-1):
        for l in range (k+1,p):
            matrice_covariance[k][l] = matrice_covariance[l][k]
    return matrice_covariance

def new_matrice(M,l):
    new_M = M[::]
    for k in range(len(new_M)):
        new_M[k]=new_M[k][:l]+new_M[k][l+1:]
    return new_M[1:]

def determinant_matrice(M):
    l = len(M)
    if l == 1 : return M[0][0]
    else :
        sum = 0
        signe = 1
        # developpement sur la première ligne
        for k in range(l):
            new_M = new_matrice(M,k)
            sum += M[0][k] * signe * determinant_matrice(new_M)
            signe = -signe
        return sum

def eigen(M):
    matrice = np.array(M)
    eigen = np.linalg.eig(matrice)
    return eigen

def tri_eigen(Eigen):
    eigen_values , eigen_vectors =  Eigen
    eigen_values = eigen_values.tolist()
    eigen_vectors = eigen_vectors.tolist()
    sorted_values = []
    sorted_vectors = []
    for k in range(len(eigen_values)):
        m = eigen_values.index(max(eigen_values))
        sorted_values.append(eigen_values.pop(m))
        sorted_vectors.append(eigen_vectors.pop(m))
    return sorted_values,sorted_vectors


def dot_product (a,b):
    res = 0
    l = len(a)
    for k in range (len(a)):
        res += a[k]*b[k]
    return res

def mult_vect (scalaire,vect):
    return list(map(lambda x : scalaire*x , vect))

def add_vect (vect1,vect2):
    return list(map(lambda x,y : x+y , vect1, vect2))


def projection (eigen_vectors,dimension,dataset):
    data = dataset[::]
    vecteurs = eigen_vectors[:dimension]
    for k in range (len(dataset)):
        new_point = [0 for k in range (dimension)]
        for i in range (dimension):
            eigen_i = vecteurs[i]
            dot_prod = dot_product(dataset[k],eigen_i)
            new_point = add_vect(new_point,mult_vect(dot_prod,eigen_i))
        data[k] = new_point
    return data

def PCA_maison(dataset,dimension):
    M_cov = matrice_covariance(dataset)
    vectors = tri_eigen(eigen(M_cov))[1]
    return projection(vectors,dimension,dataset)

def trace_PCA_2D(dataset):
    data = PCA_maison(dataset,2)
    x = [data[k][0] for k in range (len(dataset))]
    y = [data[k][1] for k in range (len(dataset))]
    plt.scatter(x,y)
    plt.show()

def trace_couleur_pca_2D(dataset):
    data = datatest(dataset)
    pca = PCA_maison(data,2)
    rep = reponse_test(dataset)
    for k in range (len(dataset)):
        if rep[k] == 'B':
            plt.scatter(pca[k][0],pca[k][1],c='b')
        else : plt.scatter(pca[k][0],pca[k][1],c='r')
    plt.show()

def trace_couleur_pca_1D(dataset):
    data = datatest(dataset)
    pca = PCA_maison(data,1)
    rep = reponse_test(dataset)
    for k in range (len(dataset)):
        if rep[k] == 'B':
            plt.scatter(k,pca[k][0],c='b')
        else : plt.scatter(k,pca[k][0],c='r')
    plt.show()

def trace_couleur_pca_3D(dataset):
    data = datatest(dataset)
    pca = PCA_maison(data,3)
    rep = reponse_test(dataset)
    ax = plt.axes(projection='3d')
    for k in range (len(dataset)):
        if rep[k] == 'B':
            ax.scatter3D(pca[k][0],pca[k][1],pca[k][2],c='b')
        else : ax.scatter3D(pca[k][0],pca[k][1],pca[k][2],c='r')
    plt.show()

from mpl_toolkits import mplot3d
import numpy as np

def visualisation_groupe_PCA_maison(dataset,k,liste):
    a = datatest(dataset)
    c = 0
    d = regroupement_n(datatest(dataset),k)[0]
    x = res(dataset,d,k)
    seuil = liste[k-2]
    while x > seuil and c < 400:
        d = regroupement_n(datatest(dataset),k)[0]
        x = res(dataset,d,k)
        c += 1
    e = reponse_test(dataset)
    pca = PCA_maison(a,2)
    Y = pca
    fig1 = plt.figure(1)
    ax1 = fig1.gca()
    fig2 = plt.figure(2)
    ax2 = fig2.gca()
    for i in range (len(Y)) :
        if e[i] == 'M' :
            ax2.scatter(Y[i][0],Y[i][1],c='r')
        else : ax2.scatter(Y[i][0],Y[i][1],c='b')
    for k in range (len(Y)):
        ax1.scatter(Y[k][0],Y[k][1],c=couleurs[d[k]])
    plt.show()
    return x

## Opti temps
import time as t

def courbe_temps (dataset,liste,n):
    temps = []
    precision = liste
    for k in range (2,len(liste)+2):
        t1 = t.time()
        u = sortie_algo(dataset,liste,k,n)
        t2 = t.time()
        temps.append(t2-t1)
    plt.plot(liste,temps,'b')
    plt.show()
    return temps

## Application du clustering à un cas

# le clustering detecte les groupes dans le dataset ,
# le dataset reste le meme pour un hopital, il suffit de définir le nombre de groupes et on peut obtenir
# une liste des centroides :
# [[coord.centroide1 + classe du groupe],[coord.centroide2 + classe du groupe],...]
# Une fois qu'on a obtenu cette liste il suffit de trouver quel est le centroide le plus proche du patient
# On connait alors le groupe auquel appartient le patient et donc sa classe
# On obtient alors une classification rapide du patient puisqu'on le compare seulement aux n centroides et
# non pass à l'entiereté du dataset
# Le calcul de centroides et de leur classe étant fait à l'avance, et ne changeant que si le dataset change
# Pour déterminer la classe d'un centroides on peut éffectuer un KNN par exemple ou étudier les classes des # éléments connus du groupe


def classe_centroides(dataset,k,liste):
    c = 0
    [d,A] = regroupement_n(datatest(dataset),k)
    x = res(dataset,d,k)
    seuil = liste[k-2]
    while x > seuil and c < 400:
        [d,A] = regroupement_n(datatest(dataset),k)
        x = res(dataset,d,k)
        c += 1
    classe_centroides = traitement2(d,dataset,k)[1]
    return [classe_centroides,A]

from math import sqrt
def distance(a,liste):
    t = []
    for k in range(len(liste)):
        res = 0
        for l in range(len(a)):
            res += (a[l]-liste[k][l])**2
        t.append(sqrt(res))
    return t

# On a alors un raisonnement individuel, bien sur cela n'est pas rapide codé comme ca,
# Pour rendre se résonnement rapide il faudrait se fixer sur une valeur de k et calculer à l'avance
# classe_centroides pour qu'il ne reste plus que classe_rapide_patient


def classe_patient(patient,dataset,k,liste):
    [classe_centroide,A] = classe_centroides(dataset,k,liste)
    liste_distance_patient_centroides = distance(patient,A)
    groupe_patient = liste_distance_patient_centroides.index(min(liste_distance_patient_centroides))
    classe_patient = classe_centroide[groupe_patient]
    return classe_patient

def classe_patient_rapide (patient,centroides):
    [classe_centroide,coord_centroides] = centroides
    distance_patient = distance(patient,coord_centroides)
    groupe_patient = distance_patient.index(min(distance_patient))
    classe_patient = classe_centroide[groupe_patient]
    return classe_patient



## Interface utilisateur avec classe_patient_rapide

from tkinter import *


def interface_mass_clustering():


    def onclick_mass():
        patient = [bi_rads.get(),age.get(),forme.get(),marge.get(),densite.get(),'?']
        data = base_de_données_mass[::]
        data.append(patient)
        stand_data = standardization(data)
        stand_patient = stand_data[-1][:-1]


        predict = classe_patient_rapide(stand_patient,classe_centroides_mass_6)


        window2 = Tk()
        window2.title("Breast Cancer Predictor")
        window2.geometry("550x400")
        window2.minsize(550,400)
        window2.iconbitmap("logo_predictor.ico")
        window2.config(background='#F6C0DD')
        if predict == 'B' :
            Label(window2, text="Le diagnostic du patient est négatif", font=("arial", 20), fg="white", bg="#F6C0DD", height=2).pack(expand = YES)
        else :
            Label(window2, text="Le diagnostic du patient est positif", font=("arial", 20), fg="white", bg="#F6C0DD", height=2).pack(expand = YES)



    # créer fenetre
    window = Tk()

    # pesonnalise fenetre
    window.title("Breast Cancer Predictor Clustering Mass")
    window.geometry("1080x720")
    window.minsize(480,360)
    window.iconbitmap("logo_predictor.ico")
    window.config(background='#F6C0DD')

    # créer frame

    frame_globale = Frame(window,bg='#F6C0DD')
    frame_données = Frame(frame_globale,bg = '#C56D9C')


    # ajout texte global
    texte_globale = Label(frame_globale, text = "Veuillez rentrer les données",pady = 15, font = ('arial', 20) , bg = '#F6C0DD' ,fg = '#FFFFFF')
    texte_globale.pack()

    # ajout texte frame données
    texte_donnée1 = Label(frame_données, text = "BI-Rades",pady = 10,padx = 50, font = ('arial', 14) , bg = '#C56D9C', fg = '#FFFFFF').grid(row = 0 , column = 0)
    bi_rads = DoubleVar()
    Entry(frame_données,textvariable= bi_rads,width=10, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=0,column=1)

    texte_donnée2 = Label(frame_données, text = "Age",pady = 10,padx = 50, font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 1 , column = 0)
    age = DoubleVar()
    Entry(frame_données,textvariable= age,width=10, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=1,column=1)

    texte_donnée3 = Label(frame_données, text = "Forme",pady = 10,padx = 50, font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 2 , column = 0)
    forme = DoubleVar()
    Entry(frame_données,textvariable= forme,width=10, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=2,column=1)

    texte_donnée4 = Label(frame_données, text = "Marge",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 3 , column = 0)
    marge = DoubleVar()
    Entry(frame_données,textvariable= marge,width=10, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=3,column=1)

    texte_donnée5 = Label(frame_données, text = "Densité",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 4 , column = 0)
    densite = DoubleVar()
    Entry(frame_données,textvariable= densite,width=10, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=4,column=1)


    #ajouter bouton

    predict_button = Button(frame_globale,text = 'Prédiction', font = ('arial', 20) , bg = '#C56D9C', fg = '#FFFFFF',command = onclick_mass)
    predict_button.pack(pady = 20, side = BOTTOM , fill = X)




    frame_données.pack(expand=YES,fill = X)

    frame_globale.pack(expand = YES)

    window.mainloop()



def interface_wisconsin_clustering():


    def onclick_wisconsin():
        patient=[radius_mean.get(),texture_mean.get(),perimeter_mean.get(),area_mean.get(),smoothness_mean.get(),compactness_mean.get(),concavity_mean.get(),concave_points_mean.get(),symmetry_mean.get(),fractal_dimension_mean.get(),radius_se.get(),texture_se.get(),perimeter_se.get(),area_se.get(),smoothness_se.get(),compactness_se.get(),concavity_se.get(),concave_points_se.get(),symmetry_se.get(),fractal_dimension_se.get(),radius_worst.get(),texture_worst.get(),perimeter_worst.get(),area_worst.get(),smoothness_worst.get(),compactness_worst.get(),concavity_worst.get(),concave_points_worst.get(),symmetry_worst.get(),fractal_dimension_worst.get(),'?']

        data = Base_de_données_wisconsin[::]
        data.append(patient)
        stand_data = standardization(data)
        stand_patient = stand_data[-1][:-1]

        predict = classe_patient_rapide(stand_patient,classe_centroides_wis_10)

        window2 = Tk()
        window2.title("Breast Cancer Predictor")
        window2.geometry("550x400")
        window2.minsize(550,400)
        window2.iconbitmap("logo_predictor.ico")
        window2.config(background='#F6C0DD')
        if predict == 'B' :
            Label(window2, text="Le diagnostic du patient est négatif", font=("arial", 20), fg="white", bg="#F6C0DD", height=2).pack(expand = YES)
        else :
            Label(window2, text="Le diagnostic du patient est positif", font=("arial", 20), fg="white", bg="#F6C0DD", height=2).pack(expand = YES)


    # créer fenetre
    window = Tk()

    # pesonnalise fenetre
    window.title("Breast Cancer Predictor Clustering Wisconsin")
    width= window.winfo_screenwidth()
    height= window.winfo_screenheight()
    window.geometry('%dx%d' % (width, height))
    window.minsize(1000,800)
    window.iconbitmap("logo_predictor.ico")
    window.config(background='#F6C0DD')

    # créer frame

    frame_globale = Frame(window,bg='#F6C0DD')
    frame_données = Frame(frame_globale,bg = '#C56D9C')


    # ajout texte global
    texte_globale = Label(frame_globale, text = "                                                                 Veuillez rentrer les données                                                                   ",pady = 15, font = ('arial', 20) , bg = '#F6C0DD' ,fg = '#FFFFFF')
    texte_globale.pack()

    # ajout texte frame données
    texte_donnée1 = Label(frame_données, text = "Rayon moyen",pady = 10,padx = 50, font = ('arial', 14) , bg = '#C56D9C', fg = '#FFFFFF').grid(row = 0 , column = 0)
    radius_mean = DoubleVar()
    Entry(frame_données,textvariable= radius_mean,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=0,column=1)

    texte_donnée2 = Label(frame_données, text = "Texture moyenne",pady = 10,padx = 50, font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 1 , column = 0)
    texture_mean = DoubleVar()
    Entry(frame_données,textvariable= texture_mean,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=1,column=1)

    texte_donnée3 = Label(frame_données, text = "Périmètre moyen",pady = 10,padx = 50, font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 2 , column = 0)
    perimeter_mean = DoubleVar()
    Entry(frame_données,textvariable= perimeter_mean,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=2,column=1)

    texte_donnée4 = Label(frame_données, text = "Aire moyenne",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 3 , column = 0)
    area_mean = DoubleVar()
    Entry(frame_données,textvariable= area_mean,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=3,column=1)

    texte_donnée5 = Label(frame_données, text = "Lisseté moyenne",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 4 , column = 0)
    smoothness_mean = DoubleVar()
    Entry(frame_données,textvariable= smoothness_mean,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=4,column=1)

    texte_donnée6 = Label(frame_données, text = "Compacité moyenne",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 5 , column = 0)
    compactness_mean = DoubleVar()
    Entry(frame_données,textvariable= compactness_mean,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=5,column=1)

    texte_donnée7 = Label(frame_données, text = "Concavité moyenne",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 6 , column = 0)
    concavity_mean = DoubleVar()
    Entry(frame_données,textvariable= concavity_mean,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=6,column=1)

    texte_donnée8 = Label(frame_données, text = "Nombre moyen de creux",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 7 , column = 0)
    concave_points_mean = DoubleVar()
    Entry(frame_données,textvariable= concave_points_mean,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=7,column=1)

    texte_donnée9 = Label(frame_données, text = "Symétrie moyenne",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 8 , column = 0)
    symmetry_mean = DoubleVar()
    Entry(frame_données,textvariable= symmetry_mean,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=8,column=1)

    texte_donnée10 = Label(frame_données, text = "Dimension fractale moyenne",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 9 , column = 0)
    fractal_dimension_mean = DoubleVar()
    Entry(frame_données,textvariable= fractal_dimension_mean,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=9,column=1)

    texte_donnée11 = Label(frame_données, text = "Ecart-type Rayon",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 0 , column = 2)
    radius_se = DoubleVar()
    Entry(frame_données,textvariable= radius_se,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=0,column=3)

    texte_donnée12 = Label(frame_données, text = "Ecart-type Texture",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 1 , column = 2)
    texture_se = DoubleVar()
    Entry(frame_données,textvariable= texture_se,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=1,column=3)

    texte_donnée13 = Label(frame_données, text = "Ecart-type Perimetre",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 2 , column = 2)
    perimeter_se = DoubleVar()
    Entry(frame_données,textvariable= perimeter_se,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=2,column=3)

    texte_donnée14 = Label(frame_données, text = "Ecart-type Aire",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 3 , column = 2)
    area_se = DoubleVar()
    Entry(frame_données,textvariable= area_se,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=3,column=3)

    texte_donnée15 = Label(frame_données, text = "Ecart-type Lisseté",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 4 , column = 2)
    smoothness_se = DoubleVar()
    Entry(frame_données,textvariable= smoothness_se,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=4,column=3)

    texte_donnée16 = Label(frame_données, text = "Ecart-type Compacité",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 5 , column = 2)
    compactness_se = DoubleVar()
    Entry(frame_données,textvariable= compactness_se,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=5,column=3)

    texte_donnée17 = Label(frame_données, text = "Ecart-type Concavité",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 6 , column = 2)
    concavity_se = DoubleVar()
    Entry(frame_données,textvariable= concavity_se,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=6,column=3)

    texte_donnée18 = Label(frame_données, text = "Ecart-type Nombre points concaves",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 7 , column = 2)
    concave_points_se = DoubleVar()
    Entry(frame_données,textvariable= concave_points_se,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=7,column=3)

    texte_donnée19 = Label(frame_données, text = "Ecart-type Symétrie",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 8 , column = 2)
    symmetry_se = DoubleVar()
    Entry(frame_données,textvariable= symmetry_se,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=8,column=3)

    texte_donnée20 = Label(frame_données, text = "Ecart-type Dimension fractale",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 9 , column = 2)
    fractal_dimension_se = DoubleVar()
    Entry(frame_données,textvariable= fractal_dimension_se,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=9,column=3)

    texte_donnée21 = Label(frame_données, text = "Pire Rayon",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 0 , column = 4)
    radius_worst = DoubleVar()
    Entry(frame_données,textvariable= radius_worst,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=0,column=5)

    texte_donnée22 = Label(frame_données, text = "Pire Texture",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 1 , column = 4)
    texture_worst = DoubleVar()
    Entry(frame_données,textvariable= texture_worst,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=1,column=5)

    texte_donnée23 = Label(frame_données, text = "Pire Périmètre",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 2 , column = 4)
    perimeter_worst = DoubleVar()
    Entry(frame_données,textvariable= perimeter_worst,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=2,column=5)

    texte_donnée24 = Label(frame_données, text = "Pire Aire",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 3 , column = 4)
    area_worst = DoubleVar()
    Entry(frame_données,textvariable= area_worst,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=3,column=5)

    texte_donnée25 = Label(frame_données, text = "Pire Lisseté",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 4 , column = 4)
    smoothness_worst = DoubleVar()
    Entry(frame_données,textvariable= smoothness_worst,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=4,column=5)

    texte_donnée26 = Label(frame_données, text = "Pire Compacité",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 5 , column = 4)
    compactness_worst = DoubleVar()
    Entry(frame_données,textvariable= compactness_worst,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=5,column=5)

    texte_donnée27 = Label(frame_données, text = "Pire Concavité",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 6 , column = 4)
    concavity_worst = DoubleVar()
    Entry(frame_données,textvariable= concavity_worst,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=6,column=5)

    texte_donnée28 = Label(frame_données, text = "Pire Nombre points concaves",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 7 , column = 4)
    concave_points_worst = DoubleVar()
    Entry(frame_données,textvariable= concave_points_worst,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=7,column=5)

    texte_donnée29 = Label(frame_données, text = "Pire Symétrie",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 8 , column = 4)
    symmetry_worst = DoubleVar()
    Entry(frame_données,textvariable= symmetry_worst,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=8,column=5)

    texte_donnée30 = Label(frame_données, text = "Pire Dimension fractale",pady = 10,padx = 50,font=('arial', 14),bg = '#C56D9C', fg = '#FFFFFF').grid(row = 9 , column = 4)
    fractal_dimension_worst = DoubleVar()
    Entry(frame_données,textvariable= fractal_dimension_worst,width=7, font = ('arial',15) , bg = '#F6C0DD',fg = '#FFFFFF').grid(row=9,column=5)




    #ajouter bouton

    predict_button = Button(frame_globale,text = 'Prédiction', font = ('arial', 20) , bg = '#C56D9C', fg = '#FFFFFF',command = onclick_wisconsin)
    predict_button.pack(pady = 20, side = BOTTOM , fill = X)


    # affichage

    frame_données.pack(expand=YES,fill = X)

    frame_globale.pack(expand = YES)

    window.mainloop()




## Classes centroides calculés à l'avance

classe_centroides_wis_4 = [['M', 'M', 'B', 'B'], [[0.8343210583289392, 0.466959807972007, 0.8317736532090727, 0.7706893210201486, 0.3865277769296419, 0.5307725282270463, 0.666608195779516, 0.8126244672798042, 0.31509265373881495, -0.17898312682560885, 0.4523071072613145, -0.12327946674629217, 0.4030571446488874, 0.4389399063291649, -0.1914319670284355, 0.16149698497818157, 0.16188245489841294, 0.3389891589283735, -0.23096505559416183, -0.0478798825383128, 0.9191942525027779, 0.587031464431845, 0.901741594398296, 0.8373382350085397, 0.6002792099240015, 0.6523456523815038, 0.7428595097539039, 0.9448497740421119, 0.49317344161498244, 0.3546101318328785], [1.4160014047062472, 0.6329147718681187, 1.5147255225956158, 1.510549688896862, 0.943370173196557, 1.9439960299692571, 2.109964483368514, 1.93312337702595, 1.0786234959367191, 0.8604569657251196, 1.778639348119235, 0.37227363057220336, 1.8683445469196114, 1.6808563522876068, 0.3781643375921468, 1.6707562020641291, 1.5457738501207663, 1.56892371920451, 0.8808368451406792, 1.2400002992777321, 1.421329161930155, 0.4489963615492503, 1.5363725848403458, 1.4656145078255054, 0.5515333808477165, 1.5020304364910668, 1.651496114257827, 1.60560633094417, 0.7839702397729793, 1.0283183522222612], [-0.8766283466982466, -0.32562804487581076, -0.8463307868765428, -0.7855660613754745, 0.4809610559589319, -0.0049816716268585096, -0.32938223693974283, -0.45215044710374047, 0.28482414864320366, 0.86567100230133, -0.32357124881965593, 0.3983500427048232, -0.2805195864876616, -0.41684255793588676, 0.9408941806173666, 0.31669771965522203, 0.12562521166950472, 0.16788417745055287, 0.48340441864552874, 0.5405575305062554, -0.8516405461963373, -0.36703197117669617, -0.8183014715177819, -0.7451149820036028, 0.30670305880176063, -0.179150000840664, -0.33350794829347463, -0.49001508397606847, -0.11207511569979171, 0.291022741844419], [-0.3655155711038566, -0.24648693713884717, -0.39889454339021124, -0.3916353997073805, -0.6372074705911543, -0.7070329473508482, -0.6695111073585122, -0.6576170936050019, -0.5357974261038972, -0.47903371091117997, -0.4818376667643382, -0.19247359726526544, -0.4929685709780126, -0.4108394915690324, -0.4044342042240848, -0.589680537109817, -0.47576128910716153, -0.5993716836294734, -0.27880589674180195, -0.4837585255970623, -0.4257043488123699, -0.2558767149401408, -0.45557606044217236, -0.4380552868014787, -0.594145048881235, -0.6017040069126147, -0.6132912553505684, -0.6450372342502246, -0.3911584571051336, -0.5501494143839275]]]


classe_centroides_wis_10 = [['B', 'M', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'M'], [[-0.9318331955472988, -0.22290599025738558, -0.8427843706920922, -0.7823983688063418, 0.21762065492609922,0.9881560764731072, 1.0372855317862655, -0.009552087907530413, 0.8400355379876554, 2.440785909038775, 0.02519547986512775, 0.6508896552704976, 0.04370809138932788, -0.2654342477774734, 1.8215710802709697, 2.7467461204714976, 3.13976883292884, 2.1757628004463165, 0.8988363913072553, 3.596116372116104, -0.9171712870395626, -0.45867151656303784, -0.8462167473063502, -0.7653138321249413, 0.016899771625683308, 0.4925196700336733, 0.9175675158520284, -0.067261966252537, -0.13844843712989516, 1.4834524452015152], [2.0332361804673185, 0.7963790081701753, 2.1197877114434625, 2.21216631634832, 0.8909713218632273, 1.9701771478218855, 2.1677091815375027, 2.2985028826462717, 1.0873013388785246, 0.4404857328386664, 2.438941611671572, 0.2608841754722837, 2.5467292432111797, 2.4687869076472166, 0.12037227064620991, 1.2271059891238827, 0.9719869215226227, 1.2787072459178914, 0.5605442073208027, 0.6337402597820818, 2.11722285249506, 0.5896826389160144, 2.223822878012082, 2.2691528887363726, 0.4970777072944668, 1.3606157496957705, 1.4620421983044247, 1.80705543369865, 0.6582635041875722, 0.6379630102757755], [1.3219423646387984, 0.4767704145613412, 1.2840805604838303, 1.2848001725512213, 0.11209944800754638, 0.34504717803709317, 0.67961365956337, 0.9742269661195518, 0.10740011268088619, -0.6799086029488001, 0.8435216764463733, -0.04301264803940707, 0.7358160040671182, 0.8017410830903187, -0.21927567002432222, 0.02081175908260343, 0.1217209418548031, 0.4694299156574436, -0.17980604134769895, -0.18246990991035866, 1.3451688092302965, 0.47772019698610646, 1.2771593560128867, 1.2820712137018706, 0.22938596663243432, 0.26040351346484, 0.544437099867231, 0.9298171091521606, 0.18894368315337942, -0.22153043467896977], [-0.036954314156168025, 0.9394662472528514, -0.08139055365910691, -0.10883850547102551, -0.7454206243307022, -0.6505893162981589, -0.5255572828998263, -0.4824714653453578, -0.5463669187877926, -0.7672480381880933, -0.28896193986553886, 0.2931156705348065, -0.3002893375044415, -0.252971364739937, -0.4501510061245247, -0.4483482363682786, -0.33878905522284575, -0.3475669097564613, -0.36210386358302976, -0.504514322736562, -0.10771554116535563, 0.9594970560170387, -0.14583779898359478, -0.1781574770330375, -0.6331973104609738, -0.4816411534922964, -0.42626352213342916, -0.38796531159877307, -0.3547598259185549, -0.6397047052626772], [-0.22176382765280236, -0.6645221122634789, -0.2196469740203777, -0.29234275846054153, 0.3421572712565298, -0.021160079862290154, -0.2969653015544315, -0.21070763921124008, -0.011893014065963301, 0.10553686422696278, -0.5202219101977482, -0.7968635376326837, -0.4981050555090383, -0.41887181604644524, -0.38505950801530114, -0.30279399704361415, -0.25673094961803666, -0.2877866886764998, -0.3596965061318594, -0.2734656251387728, -0.26962049365916374, -0.6585902094574714, -0.2562150149459026, -0.3351383733008071, 0.3104197429228233, 0.02057000505708799, -0.08490489211234063, -0.02138661681653265, 0.1451191619482757, 0.10768460138755045], [-0.33060129193464155, -0.7786256893174699, -0.3804815230985946, -0.3735511623276588, -0.9471160131373432, -0.9272815801929769, -0.7991921926505595, -0.787076573051773, -0.8975333959880537, -0.7031317750089981, -0.5897342926257113, -0.5927996998224401, -0.6018740835757075, -0.45851067274814244, -0.581744817426182, -0.7489397380000975, -0.614485520296801, -0.7795472312703551, -0.4615097819991123, -0.6069406880112128, -0.42642223491636877, -0.779091665064059, -0.4725884464693305, -0.44431717269718257, -0.9179524632707973, -0.7733511841540124, -0.778042469699711, -0.8124600441769215, -0.6891310271374899, -0.7332688336472861], [-0.5510099702573812, -0.18603406307217787, -0.5167832678741628, -0.5478157885514259, 0.05594328085207172, 0.19516545729244048, -0.09573241420416344, -0.2808030910816724, -0.16888142241760667, 0.5817977876805086, -0.3062058882468856, 0.3659141336021095, -0.19685454309704883, -0.35421709407004565, 0.6213584635526852, 0.7959710608870398, 0.5333521121209543, 0.6177826110090979, 0.2747244397810233, 0.745481231012467, -0.6066985923092206, -0.28923652584139015, -0.5464414822240006, -0.5792624231687196, -0.12145705470099262, 0.0998256071461582, -0.048543661711703004, -0.22037178707939678, -0.4652846652792524, 0.30605636080271253], [-1.0624852946837933, 0.4756567467222498, -1.0657154261958055, -0.9115892205286998, 0.047732000177517046, -0.7287346208619677, -0.7765942135832453, -0.8240105838304744, -0.18139820283360847, 0.28310358579767336, -0.06437166675614739, 2.158669912030385, -0.13099791760033855, -0.327900552693089, 1.4381764508386252, -0.4056000103523333, -0.5055799329495848, -0.5522217866612456, 0.8237819636295108, 0.07784266372014824, -1.0026155691382839, 0.33694173927492604, -1.012982984340909, -0.8377388017570266, -0.17791208668494604, -0.8629746961568256, -0.9355130097873696, -1.1307254448540545, -0.6535117183718253, -0.40147746880406027], [-0.9667815886593363, -0.4668936342754692, -0.9722463981143996, -0.8484661162328669, -0.06231414480650914, -0.7073709383070417, -0.8020196467322648, -0.8294350583141055, 0.1330763902484622, 0.14029324681285996, -0.46794446697286907, 0.006598739532321772, -0.4887669378381871, -0.485008632899944, 0.26812748995817093, -0.6235964443385126, -0.5527104457301021, -0.6546626715288848, 0.30215254975549405, -0.36033353586187, -0.9114861123440862, -0.44167986261037195, -0.9254086751377514, -0.7836478276356836, -0.039853048607768525, -0.7536136517840952, -0.837077443960792, -0.9168944442133249, 0.00028439786273122607, -0.36120838840629893], [0.11979695197758794, 0.4375281044411343, 0.20239603486234922, 0.013160402105503333, 1.1476586013973826, 1.3239661571506198, 1.0710868709369377, 0.8953592969189156, 1.044300940172682, 1.0527423074711038, -0.02572796444139521, -0.10556309641404286, 0.035189858774199154, -0.07749376153612722, 0.004189017899573967, 0.8170058511824639, 0.48919606452346537, 0.4650387736131935, 0.18560358848961026, 0.4815365665450127, 0.26753014879687065, 0.7488230344632145, 0.35304520874098066, 0.1268071275056975, 1.406675683895577, 1.730013486605323, 1.4572376022608715, 1.2314601231050641, 1.4277483310910213, 1.694222802661669]]]




classe_centroides_mass_4 = [['M', 'B', 'M', 'B'], [[0.24962647704976082, 1.0278461037764457, 0.6806121493995173, 0.7045028121194713, 0.27440941267484037], [-0.21398746621328757, -0.6123802900820656, -1.011275356641681, -1.0548057943776659, 0.2498451533435082], [0.044660591227929675, -0.2852697738019115, 0.7164634836375205, 0.810144323857553, 0.2955625087208993], [-0.10594663987882616, -0.20601144462270155, -0.1849924528742332, -0.3667704806240104, -3.078888625607654]]]

classe_centroides_mass_6 = [['B', 'B', 'M', 'M', 'B', 'M'], [[-0.10594663987882624, -0.2060114446227014, -0.184992452874233, -0.36677048062401024, -3.0788886256076515], [-0.22379037233084473, -0.6226203441266367, -1.0181899893499973, -1.1441334074972171, 0.25057685468529206], [0.15011222339419925, 1.0586292084693856, 0.8785347583872962, 0.7083723618526774, 0.27934662579003], [0.05617963744271581, 0.848417409302648, -1.0100978648399046, 0.6512965032878808, 0.2404660725079085], [-0.21839253687700452, -0.4352419928513721, -0.3515374045761005, 0.513927487759052, 0.24046607250790855], [0.2774808429573732, -0.2948319950464843, 0.952614734588175, 0.8215566915489654, 0.3075540860143153]]]

