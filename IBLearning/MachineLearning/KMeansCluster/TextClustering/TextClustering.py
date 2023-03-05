"""
    TextClustering.py
    IBLearning project
    By Alix Hamidou
    2022

    Ce fichier contient la classe TextClustering
"""

from .ClusterType import ClusterType
from .Cluster import Cluster
from .PreProcessing import Language


class TextClustering:
    """
        Permet la création d'un cluster de texte

        Les textes peuvent êtres de tailles différentes

        Fonctionnement du cluster de texte:
            - Les textes sont convertis en liste de mots
            - Entre deux textes, les mots en commun sont comptés
            - Plus le nombre de mots en commun est élevé, plus les textes sont proches

        Fonctionnement du module:
            - On crée l'objet TextCluster
            - Lors de l'ajout d'un texte, on compare le nouveau texte avec les autres textes déjà présents
            - Quand on a comparé le nouveau texte avec tous les autres textes, on ajoute le nouveau texte au cluster le plus proche
            - Le problème est que si on a pas de texte de base, on ne peut pas comparer le nouveau texte avec les autres textes
            - Pour résoudre ce problème, on crée un cluster de base avec un texte de base
            - Puis au bout de X textes, on regarde la moyenne de ressemblance entre les textes, les textes les plus proches sont fusionnés, les textes les moins proches sont séparés et forme un nouveau cluster

        Fonction:
            - AddText: Ajout d'un texte au cluster le plus proche
            - RebaseClusterMean: Recalcule la moyenne de ressemblance entre les textes du cluster pour déterminer si le cluster est précis ou non et créé un nouveau cluster si besoin

    """
    clusters: list[Cluster]
    language: str

    def __init__(self, language: str) -> None:
        """
            Création d'un cluster de texte

            On initialise la liste des clusters et la langue du cluster
            - from IBLearning.MachineLearning.KMeansCluster.TextClustering.PreProcessing import Language
            - Language.FRENCH
        """
        self.clusters = []
        self.language = language
        return
    
    def AddText(self, text: str) -> None:
        """
            Ajout d'un texte au cluster le plus proche

            Compare le texte avec les clusters existants

            Si aucun cluster n'existe, on crée un nouveau cluster avec le texte

            Sinon, on récupère le cluster le plus proche et on ajoute le texte au cluster
        """
        if len(self.clusters) == 0:
            self.clusters.append(Cluster(ClusterType.UNIGRAM, self.language))
            self.clusters[-1].AddText(text)
            return
        
        # On compare le texte avec les clusters
        result = self.__CompareTextToClusters(text)

        # On récupère le cluster le plus proche
        closestCluster = self.clusters[result.index(max(result))]
        closestCluster.AddText(text)
        return

    def __CompareTextToClusters(self, text: str) -> list[float]:
        """
            Analyse d'un texte pour le comparer avec les clusters

            Permet de savoir dans quel cluster le texte doit être ajouté

            Retourne une liste de pourcentage de ressemblance entre le texte et les clusters [cluster1, cluster2, ...]
        """
        
        result: list[float] = []
        for cluster in self.clusters:
            result.append(cluster.CompareText(text))
        return result



    def RebaseClusterMean(self, accuracyMin: float) -> None:
        """
            Recalcule la moyenne de ressemblance entre les textes du cluster pour déterminer si le cluster est précis ou non et créé un nouveau cluster si besoin

            Si la moyenne de ressemblance est inférieur à accuracyMin, on prend tout les textes les plus éloignés du cluster et on les ajoute dans un nouveau cluster, et le reste des textes sont ajoutés dans un second cluster, puis le cluster de base est supprimé
        """
        newClusters: list[Cluster] = []
        for cluster in self.clusters:
            if cluster.BaseMean() < accuracyMin:
                # On récupère les textes les plus éloignés du cluster
                texts: list[list[str,float]] = cluster.GetTextsMean()

                # On crée un nouveau cluster avec les textes les plus éloignés
                newCluster = Cluster(ClusterType.UNIGRAM, self.language)

                # On crée un nouveau cluster avec les textes les plus proches
                newClusterAccurate = Cluster(ClusterType.UNIGRAM, self.language)

                for text in texts:
                    if text[1] < accuracyMin:
                        newCluster.AddText(text[0])
                    else:
                        newClusterAccurate.AddText(text[0])

                # On supprime le cluster de base
                self.clusters.remove(cluster)

                # On ajoute les nouveaux clusters
                newClusters.append(newCluster)
                newClusters.append(newClusterAccurate)
            else:
                newClusters.append(cluster)
                
        self.clusters = newClusters
        return
    
