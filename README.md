# KNN
Il s'agit d'un projet école, réalisé en BAC+3, dans le but de maitriser la compréhension du KNN.
Le K-NN (K Nearest Neighbors en anglais) est un algorithme d’apprentissage supervisé. Il peut être à la fois utilisé dans des problèmes dit de régression ou de classification.

Le K-NN effectue une prédiction pour une valeur donnée sans calculer un modèle prédictif d’un training Set. En effet, cet algorithme ne réalise par réellement un apprentissage mais se base plutôt sur le jeu de données fourni (training Set) pour produire un résultat. Pour effectuer sa prédiction sur une donnée, l’algorithme va calculer la distance avec tous les points aux alentours et sélectionner les plus proches : les voisins (ou neighbors). Ensuite, pour ces K voisins, l’algorithme étudiera les valeurs de sortie pour déterminer celle de la donnée étudiée.
Il existe plusieurs manières de calculer la distance : euclidienne ou Manhattan par exemple. La distance euclidienne est la plus souvent utilisée et est considérée comme « la plus sûre ». 

Environnement : Spyder
Bibliothèques : NUmpy/Pandas/matplotlib.pyplot

# Fonctionnement

1. Chargement des données des datasets
2. Découpage des sets en entrainement et tests
3. Initialisation de la valeur de k
4. Calcul des distances et classement par ordre croissant
5. Récupérer les k premiers voisins
6. Retourne la classe la plus présente
