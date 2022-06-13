# Machine-learning-and-breast-cancer
## Analyse et implémentation d'algorithmes de classification : K-plus-proches-voisins et k-moyennes

__Problématique de l'étude :__  
  
  Comment mettre au point des algorithmes de classification appropriés au diagnostic médical ?  
  C’est-à-dire optimiser leurs performances et étudier leur capacité à expliquer leurs résultats.
  
__Objectif de l'étude :__   
  
   Lors de cette étude, on cherchera tout d’abord à optimiser les données d’entrées vis-à-vis du problème de classification. Une fois les algorithmes implémentés, il faudra les optimiser selon des critères à définir. Enfin, un aspect important sera de mettre au point des algorithmes annexes capable d’argumenter la sortie de l’algorithme principal et donc d’expliquer la prédiction à un humain.

__Plan de l'analyse :__  
  I.   Exploration et modification des données  
  II.  Étude et amélioration des algorithmes  
  III. Mise au point d'une analyse en composante principale  
  IV.  Mise au point du compte rendu explicatif
 
Lors de cette étude, on utilisera majoritairement la base de données Breast Cancer Wisconsin Data Set, puis pour une comparaison finale  la base Mammographic Mass Data Set. Breast Cancer Wisconsin Data Set contient 579 patients, définis sur 30 variables et 
Mammographic Mass Data Set contient 830 patients définis sur 5 variables
  
## I.   Exploration et modification des données 

Avant de travailler avec la base de données, on va l’analyser. On regarde tout d'abord si certaines données sont redondantes. Pour ce faire on établit une matrice de covariance entre les variables. On pourra ainsi lors de l’étude éliminer les données redondantes pour plus d’efficacité, même si dans une optique de diagnostic médical, garder l’intégralité des données semble meilleur car la précision prime sur la rapidité. Dans le cadre d'une campagne de dépistage en revanche, on pourrait chercher une telle efficacité.
  
![matrice correlation wis](https://user-images.githubusercontent.com/83364235/173245797-40ee2cd8-f0ad-4a68-b767-182b18e02e7d.PNG)

 Pour se rendre compte si l'on peut facilement travailler sur la base de données, on peut essayer de la visualiser en utilisant une analyse en composante principale, dont je détaillerai le fonctionnement et l'implémentation dans le IV. Il est important de noter que les axes sur la figure n'ont pas de sens physique ou médical, ils sont issus de calculs mathématiques.
  
  ![PCA wis](https://user-images.githubusercontent.com/83364235/173245965-061e0c11-f41f-46b9-9901-8ba34cbb5ea5.png)


  
Si l’on trace les courbes de densités des différentes variables, on se rend compte que certaines permettent de plus facilement établir un diagnostic. On définit alors l’importance comme la facilité à établir un diagnostic selon cette variable seule. Pour le calcul de l’importance, on passe par des calculs d’intégrales. Pour la majeure partie des variables, les valeurs des patients bénins sont plus faibles. Ainsi, on trace une ligne verticale et on calcule la proportion d’aire sous la courbe bleue à gauche de la droite par rapport à l’aire totale, pareil pour la courbe rouge mais à droite de la courbe, et on les multiplie. S’il existe une droite tel que ce nombre est élevé, alors la variable est importante.  
On pourrait d’ailleurs généraliser cette méthode si nécessaire en définissant plusieurs droites et en attribuant chaque bloc à une des deux classes. Une fois les importances obtenues on pourra alors coefficienté les distances calculées dans notre espace par ces importances afin de potentiellement réaliser un meilleur diagnostic. 
  
  ![courbe_param_27](https://user-images.githubusercontent.com/83364235/173245823-fc51a0dc-b827-4e5b-a95d-c63861a34f83.png)
![courbe_param_8](https://user-images.githubusercontent.com/83364235/173245838-c80818c5-1122-4614-9fa3-16d350936aaa.png)


 ## II.  Étude et amélioration des algorithmes 
 
Maintenant que le travail sur les données a été effectué, il s’agit d’implémenter les algorithmes et de les étudier. On peut voir sur les images qui suivent une présentation du fonctionnement des algorithmes. Le K-plus-proches voisins, ou KNN, part d’un point dont on connaît les données (noir) et le compare à d’autres pour lesquels on connait le diagnostic (bleu ou rouge). La valeur du paramètre k va donc déterminer la prédiction, comme le montre l’exemple : 3 voisins => rouge et 7 voisins => bleu, si on calcul la classe comme étant celle qui domine parmi les voisins. On cherchera alors à trouver le meilleur k. L’algorithmes des k voisins établie la classe (le diagnostic) des voisins les proches, et décide de la classe du patient à partir de celles-là. On peut alors compter le nombre de cas de chaque classe, ou bien coefficienté en introduisant un poids. Par exemple, parmi les plus proches, le plus éloigné impactera moins que le plus proche. On coefficiente alors par la distance au patient. Ainsi, en plus du paramètre k, on tâchera d’optimiser la fonction f coefficientant par la distance.
 
 ![visualisation situation pca wis cercle id=64](https://user-images.githubusercontent.com/83364235/173246224-5f432feb-2fd3-46b2-ab08-40532dc07fdf.PNG)

 
L’algorithme de clustering, de son côté, va être un algorithme non labélisé, c’est-à-dire qu’il travaille avec des données sans diagnostic et établit des groupes de ressemblance. En entrée de cet algorithme de classification basé sur du clustering, on injecte une base de données dont on ne connaît pas le diagnostic, et une plus petite dont on connaît les diagnostics. On établit les groupes, puis pour chaque donnée labélisée, on détermine son groupe. Enfin, on détermine la classe d’un groupe comme la classe majoritaire des données labélisées de ce groupe. On établit alors une prédiction sur l’intégralité de la base de départ.   
	Pour obtenir les groupes, on place k centroïdes aléatoirement dans l’espace considéré, on formera alors k groupes. Le paramètre k est donc là aussi à optimiser. Pour chaque point de l’espace, on lui associe son centroïde le plus proche. On déplace les centroïdes au centre de gravité de leurs ensembles respectifs. Puis on recommence l’opération jusqu’à obtenir une certaine stabilisation. L’initialisation étant aléatoire, les groupes obtenus en fin ne sont pas toujours les mêmes, il convient alors de répéter cette opération jusqu’à obtenir des groupes permettant d’obtenir des prédictions respectant la contrainte de précision.    
  Les centroïdes étant ici initialement placés aléatoirement, l'algorithme est probabiliste et ne renverra pas forcément la meilleure partition, il s'agit donc de le lancer un certain nombre de fois jusqu'à avoir une partition qui respecte une contrainte définie à l'avance comme par exemple un seuil d'erreur.  

 ![clustering wis 4](https://user-images.githubusercontent.com/83364235/173246311-af737367-b775-42ba-a8b7-032ec5acb058.png)
 
On en déduit alors dans cet exemple que les groupes bleu clair et vert correspondent au groupe bleu, et les groupes violet et jaune au groupe rouge. On regarde alors l'évolution des partitions avec plus de groupes.  
 
![clustering wis 6](https://user-images.githubusercontent.com/83364235/173246314-cc57a79c-08f2-459d-94c6-d4be8f13025d.png)
![clustering wis 8](https://user-images.githubusercontent.com/83364235/173246322-2a8b7517-aee6-41fe-b264-d0e479ea9826.png)

 
On cherche maintenant à évaluer la précision des algorithmes de sorte à les optimiser. Pour ce faire, on peut étudier le pourcentage d’erreurs dans les prédictions, ou dans le même esprit, on crée une matrice de confusion établissant le nombre de faux positifs, faux négatifs, … pour plus d’informations sur les erreurs. On peut alors par exemple tracer la précision de l’algorithme de clustering en fonction du paramètre k. On voit que l'algorithme est plus précis lorsque le k augmente, ce qui intuitivement semble logique dans la mesure où les groupes sont formés des patients se ressemblant le plus, et tous les groupes sont affectés d'une même classe, donc plus il y a de groupes plus on a de chance d'avoir des groupes homogènes. Cependant, la précision semble atteindre une certaine limite.     
 
 ![precision clustering](https://user-images.githubusercontent.com/83364235/173246668-b431bb0a-5392-4c4c-8feb-5563a2f1735c.PNG)

 
Une des questions était le lien entre la taille de la base de données et la précision des prédictions. La courbe ci-dessous donne ce lien pour l’algorithme des plus proches voisins. Une fois encore, les résultats sont de plus en plus précis lorsque la taille augmente, mais atteignent une limite. On remarque d’ailleurs que même si la différence semble peu significative, coefficienter l'attribution des classes par la distance semble rendre les résultats plus précis.   


![precision knn sur taille base](https://user-images.githubusercontent.com/83364235/173246812-3c136137-b6a7-4204-91cf-d7a712acf450.PNG)

Les résultats de l’algorithmes des plus proches voisins sont alors répertoriés dans ce tableau. On remarque qu’assez logiquement, la précision baisse avec moins de paramètres, elle augmente en coefficientant l'attribution des classes par la distance. Enfin, comme souhaité, on augmente en précision en coefficientant les distances par l'importance des variables.     

![precision knn](https://user-images.githubusercontent.com/83364235/173246918-eb65d69f-551f-4c8b-bd13-a9dd47500a17.PNG)


De plus, on peut analyser la précision par une courbe de ROC, cette analyse et très intéressante car elle donne une autre évaluation de la précision, mais elle est relativement limitée pour des bases de données de petite taille car l’aire sous la courbe dépend trop du moment de l’étude où apparaît l’erreur, tandis que pour une base de grande taille, il y a suffisamment d’erreurs pour que cela n’impacte pas significativement.     

![roc clustering](https://user-images.githubusercontent.com/83364235/173247021-5347200c-cf45-4ffe-b56b-01b0a08ee2d9.PNG)


Enfin, on cherche aussi à améliorer l’efficacité des algorithmes. L’algorithmes de clustering demande par exemple un temps relativement long pour une précision élevée car seules les meilleures initialisations donnent de tels résultats. Cependant, une fois obtenues, on a plus qu’à garder en mémoire les positions des centroïdes finaux. On pourrait alors recalculer les centroïdes optimaux lorsque la base de données a augmenté significativement en taille, par exemple une augmentation de 10%. On peut alors juste déterminer le groupe d’un nouveau patient pour en déduire son diagnostic, ce qui est bien plus rapide et relativement précis.     

Dans le cadre du KNN, on pourrait aussi chercher à partitionner l’espace, comme une grille. On ne considèrerait alors que les plus proches voisins des cases les plus proches, ce qui réduit le temps de calcul. En revanche, en dimension 30, selon la méthode de construction d’une telle grille, on pourrait se retrouver avec une explosion de la complexité spatiale, mais en s’intéressant à une structure pertinente pour la grille, on peut la construire en O(nln(n)) où n est le nombre de point dans l’espace, tout en ayant une complexité spatiale en O(n). Cependant, en utilisant une telle grille, si on définit µ comme étant la distance max des cases les plus proches au sens des distance de Hamming, p le nombre de partition par variable, n le nombre de patient, et d la dimension, on va devoir considérer en moyenne (2xµ + 1)^d cases où il y a en moyenne n/(p^d) patients => on passe d'un O(n) à O(n x((2xµ + 1)/p)^d ). Cependant cela suppose avoir eu accès aux points, mais cet accès aux points est en O((2xµ + 1)/p)^d) donc pour des dimension élevée comme ici et des bases de petite taille, une telle implémentation semble inutile.   

Enfin, si on projette nos données en dimension 2, puis qu’on applique la grille, on gagne en complexité spatiale car les données ont seulement 2 variables, et en complexité temporelle, mais on perd en précision lors que l’étape de projection. Il s’agit alors de trouve une projection qui conserve les distances, et c’est le but de l’étude qui suit.  

![grille](https://user-images.githubusercontent.com/83364235/173247475-c0a0aeb7-4c0a-4834-a0c6-1ee3ab2b9833.PNG)
 
 
 ## III. Mise au point d'une analyse en composante principale
 
Le but de l’analyse en composante principale est de projeter des données sur un espace de dimension inférieure, tout en gardant un maximum d’information : de variance dans les données. Cela permet alors une visualisation relativement pertinente de la situation.      
Pour ce faire on considère la matrice de covariance des variables, celle-ci est symétrique d'après le caractère symétrique de la covariance. D'après le théorème spectral, cette matrice est diagonalisable et à valeurs propres réelles. On va alors projeter les données sur les vecteurs propres ayant les plus grandes valeurs propres. On remarque qu’il n'y a pas de valeur propre de multiplicité supérieure à 1 car on travaille avec des matrices ne sortant pas d'exercice de mathématique, ce cas de figure est en fait extrêmement rare, et dans le pire des cas, il suffit de modifier d'un millième une des valeurs de la matrice pour récupérer des espaces propres de dimension 1.    
Ainsi, d’un point de vue physique ou médical, les axes n’ont pas de sens comme ils l’avaient avant, il s’agit juste des meilleurs axes de projections. On voit ci-après la comparaison entre deux algorithmes : le mien et celui de chercheurs de l'INRIA, et notamment la différence en variance. L'analyse en composantes principales va nous permettre de mieux visualiser la situation, par exemple les zones où se situent les erreurs.   

Mon algorithme :   
![pca_maison_knn_wis](https://user-images.githubusercontent.com/83364235/173247922-1d62bf1f-aa87-4fb9-9832-5e60be767e39.png)

Celui de l'INRIA :   
![PCA wis](https://user-images.githubusercontent.com/83364235/173247930-0c6bd846-04cc-4f1a-8514-3fdea2d7a2a9.png)

zones d'erreurs : 

![erreur knn](https://user-images.githubusercontent.com/83364235/173247981-91f920cc-08a7-4dbc-84e6-30c5c7e41006.PNG)



Un autre paramètre important est la conservation de la distance. En effet, le but étant de visualiser une situation au plus proche de celle en dimension élevée, il faut au maximum conserver les distances. On constate alors encore ici la différence entre les algorithmes. D’ailleurs, l’article [6] de l’annexe propose dans leur étude une réduction via des coordonnées polaire, centré en le point test, ce qui permet de conserver avec une grande efficacité les distances par rapport à ce point, mais cela s’applique difficilement dans un contexte de visualisation global des distances.      

![ecart distance](https://user-images.githubusercontent.com/83364235/173248009-bd721cb9-0a94-4750-b25a-bec9030b32e9.PNG)


## IV.  Mise au point du compte rendu explicatif

Dans l’optique d’un algorithme qui communique avec le médecin, on peut s’intéresser à la question de ce que renvoie l’algorithme. L’idée est donc ici pour le KNN de renvoyer le tableau des valeurs médicales (non transformés) des voisins, la prédiction, et de mettre un système de couleur de sorte à faciliter la lecture. Dans le cadre du clustering, on pourrait par exemple renvoyer les valeurs du patient moyen du cluster auquel appartient le patient étudié.  
  
Tableau récapitulatif KNN :   

![tableau proche couleur wis distance ( id patient = 64 ) ](https://user-images.githubusercontent.com/83364235/173248111-c73bc46c-a915-4a23-9f9e-c6e0bd60a47a.PNG)


Enfin, pour pousser l’idée d’une réelle application dans le cadre médical, on peut mettre au point une interface utilisateur entre l’algorithme et le médecin.     

 ![interface_wis](https://user-images.githubusercontent.com/83364235/173248122-a2657337-7428-4ea3-a2c2-7c7b15f202a0.PNG)

 
 ## Remarque finale : 
 
 Impact de la base de donnée :
 
 On remarque par une analyse similaire réalisée sur la deuxième base de données, de beaucoup moins bonne qualité, que sans une base de données intéressante, aucune analyse pertinente n’est possible.   
 
 ![mass](https://user-images.githubusercontent.com/83364235/173248240-4c0a1e86-ede1-4e3c-a18e-8a257f64f14f.PNG)

On peut dès l'étape de visualisation remarquer que la base semble très compacte et que la classification s'annonce complexe. Cette intuition se vérifie dans le pourcentage d'erreur du KNN : 20% d'erreurs en moyenne, et de la même manière sur la courbe de ROC.  
 


# Conclusion 

L'analyse effectuée n'est pas la même démarche que celle que pourrait avoir un chercheur visant à établir le meilleur algorithme de prédiction à ce problème donné dans la mesure où il s'agirait plutôt que de choisir des algorithmes et de les améliorer, de définir le meilleur modèle de classification.   

On peut suite à l'analyse menée, combiner ces deux algorithmes de sorte à espérer obtenir de meilleurs résultats, on se rapproche alors de la question du modèle optimal décrite précédemment.  

A travers cette étude on a pu mettre en évidence l'importance de la base de données, et notamment le format des données dans les algorithmes. La définition de l'importance et le poids par la distance dans le KNN ont amélioré la précision des algorithmes sans impacter leur complexité. Pour le clustering, on pourrait dans s'intéresser à définir les meilleures initialisations, ou du moins identifier leurs propriétés, de sorte à éviter l'aléatoire ou du moins de le limiter.   


# Références

[1] Silverman, B. W., and M. C. Jones. “E. Fix and J.L. Hodges : An Important Contribution to Nonparametric Discriminant Analysis and Density Estimation: Commentary on Fix and Hodges (1951).”   
[2] Cover, Thomas M. and Peter E. Hart : Nearest neighbor pattern classification.   
[3] Steinhaus H. : Sur la division des corps matériels en parties.   
[4] M. Emre Celebi, Hassan A. Kingravi, Patricio A. Vela : A comparative study of efficient initialization methods for the k-means clustering algorithm.   
[5] Haneen Arafat Abu Alfeilat : Effects of Distance Measure Choice on K-Nearest Neighbor Classifier Performance.  
[6] Jean-Baptiste Lamy, Boomadevi Sekar, Gilles Guezennec, Jacques Bouaud, Brigitte Séroussi : Explainable artificial intelligence for breast cancer.   
[7] William H. Wolberg and O.L. Mangasarian : Multisurface method of pattern separation for medical diagnosis applied to breast cytology.  
[8] Antoine Mazieres : Cartographie de l’apprentissage artificiel et de ses algorithmes.  


 
 
 
 
 
 
 
