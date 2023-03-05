"""
    Cluster.py
    IBLearning project
    By Alix Hamidou
    2022

    Ce fichier contient la classe Cluster qui permet de créer un cluster de texte

    Un cluster est un groupe de texte qui ont des mots en commun

    Un cluster peut être de type Unigram, Bigram ou Trigram (correpond au nombre de mots a utiliser pour comparer les textes)

    Fonction:
    - AddText: Ajout d'un texte au cluster
    - CompareText: Compare un texte avec les mots du cluster et retourne un pourcentage de ressemblance entre le texte et le cluster
    - BaseMean: Retourne la moyenne de ressemblance entre les textes du cluster pour déterminer si le cluster est précis ou non
    - GetTextsMean: Retourne la moyenne de ressemblance entre les textes du cluster
"""


from .Word import Word
from .ClusterType import ClusterType
from .PreProcessing import Preprocess


class Cluster:
    """
        Cluster de texte

        Un cluster est un groupe de texte qui ont des mots en commun

        Un cluster peut être de type Unigram, Bigram ou Trigram (correpond au nombre de mots a utiliser pour comparer les textes)

        Fonction:
        - AddText: Ajout d'un texte au cluster
        - CompareText: Compare un texte avec les mots du cluster et retourne un pourcentage de ressemblance entre le texte et le cluster
        - BaseMean: Retourne la moyenne de ressemblance entre les textes du cluster pour déterminer si le cluster est précis ou non
    """

    type: int               # Type de cluster (ClusterType, Unigram, Bigram, Trigram)
    language: str           # Langue du cluster

    texts: list[str] = None   # Liste des textes du cluster
    words: list[Word] = None  # Liste des mots du cluster

    def __init__(self, type: int, language: str) -> None:
        """
            Création d'un cluster

            On initialise le type et la langue du cluster:
            - from .PreProcessing import Language
            - Language.FRENCH
        """
        self.type = type
        self.language = language

        self.texts = []
        self.words = []
        return

    def AddText(self, text: str) -> None:
        """
            Ajout d'un texte au cluster

            On ajoute le texte puis on ajoute les mots du texte au cluster
        """
        text = Preprocess(text, self.language)

        self.texts.append(text)
        if self.type == ClusterType.UNIGRAM:
            self.__AddUnigram(text)
        elif self.type == ClusterType.BIGRAM:
            self.__AddBigram(text)
        elif self.type == ClusterType.TRIGRAM:
            self.__AddTrigram(text)
        return
    
    def __AddUnigram(self, text: str) -> None:
        """
            Ajout d'unigramme au cluster

            On ajoute les mots du texte au cluster
        """
        for word in text.split():
            self.__AddWord(word)
        return
    def __AddBigram(self, text: str) -> None:
        """
            Ajout de bigramme au cluster

            On ajoute les mots du texte au cluster
        """
        words = text.split()
        for i in range(len(words) - 1):
            self.__AddWord(words[i] + " " + words[i + 1])
        return
    def __AddTrigram(self, text: str) -> None:
        """
            Ajout de trigramme au cluster

            On ajoute les mots du texte au cluster
        """
        words = text.split()
        for i in range(len(words) - 2):
            self.__AddWord(words[i] + " " + words[i + 1] + " " + words[i + 2])
        return

    def __AddWord(self, newWord: str) -> None:
        """
            Ajout d'un mot au cluster

            Si le mot existe déjà, on incrémente le compteur,
                Sinon, on ajoute le mot au cluster
        """
        for word in self.words:
            if word.text == newWord:
                word.count += 1
                return
        self.words.append(Word(newWord, 1))
        return








    def CompareText(self, text: str) -> float:
        """
            Compare un texte avec les mots du cluster

            Retourne un pourcentage de ressemblance entre le texte et le cluster
        """
        text = Preprocess(text, self.language)

        if self.type == ClusterType.UNIGRAM:
            return self.__CompareUnigram(text)
        elif self.type == ClusterType.BIGRAM:
            return self.__CompareBigram(text)
        elif self.type == ClusterType.TRIGRAM:
            return self.__CompareTrigram(text)
        return 0.0
    
    def __CompareUnigram(self, text: str) -> float:
        """
            Compare unigramme avec les mots du cluster

            Retourne un pourcentage de ressemblance entre le texte et le cluster
        """
        count = 0
        for word in text.split():
            for clusterWord in self.words:
                if word == clusterWord.text:
                    count += 1
        return count / len(text.split())
    def __CompareBigram(self, text: str) -> float:
        """
            Compare bigramme avec les mots du cluster

            Retourne un pourcentage de ressemblance entre le texte et le cluster
        """
        count = 0
        words = text.split()
        for i in range(len(words) - 1):
            for clusterWord in self.words:
                if words[i] + " " + words[i + 1] == clusterWord.text:
                    count += 1
        return count / len(words)
    def __CompareTrigram(self, text: str) -> float:
        """
            Compare trigramme avec les mots du cluster

            Retourne un pourcentage de ressemblance entre le texte et le cluster
        """
        count = 0
        words = text.split()
        for i in range(len(words) - 2):
            for clusterWord in self.words:
                if words[i] + " " + words[i + 1] + " " + words[i + 2] == clusterWord.text:
                    count += 1
        return count / len(words)
    




    def BaseMean(self) -> float:
        """
            Retourne la moyenne du cluster

            Recalcule la moyenne de tout les textes du cluster pour savoir si il est précis
        """
        if self.type == ClusterType.UNIGRAM:
            return self.__BaseMeanUnigram()
        elif self.type == ClusterType.BIGRAM:
            return self.__BaseMeanBigram()
        elif self.type == ClusterType.TRIGRAM:
            return self.__BaseMeanTrigram()
        return 0.0

    def __BaseMeanUnigram(self) -> float:
        """
            Retourne la moyenne du cluster

            Recalcule la moyenne de tout les textes du cluster pour savoir si il est précis
        """
        mean = 0
        for text in self.texts:
            for otherText in self.texts:
                mean += self.__CompareTextsUnigram(text, otherText)
        return mean / (len(self.texts)**2)
    def __BaseMeanBigram(self) -> float:
        """
            Retourne la moyenne du cluster

            Recalcule la moyenne de tout les textes du cluster pour savoir si il est précis
        """
        mean = 0
        for text in self.texts:
            for otherText in self.texts:
                mean += self.__CompareTextsBigram(text, otherText)
        return mean / (len(self.texts)**2)
    def __BaseMeanTrigram(self) -> float:
        """
            Retourne la moyenne du cluster

            Recalcule la moyenne de tout les textes du cluster pour savoir si il est précis
        """
        mean = 0
        for text in self.texts:
            for otherText in self.texts:
                mean += self.__CompareTextsTrigram(text, otherText)
        return mean / (len(self.texts)**2)

    def __CompareTextsUnigram(self, text: str, text2: str) -> float:
        """
            Compare unigramme d'un texte avec un autre texte

            Retourne un pourcentage de ressemblance entre les deux textes
        """
        count = 0
        for word in text.split():
            for word2 in text2.split():
                if word == word2:
                    count += 1
        return count / len(text.split())
    def __CompareTextsBigram(self, text: str, text2: str) -> float:
        """
            Compare bigramme d'un texte avec un autre texte

            Retourne un pourcentage de ressemblance entre les deux textes
        """
        count = 0
        words = text.split()
        for i in range(len(words) - 1):
            words2 = text2.split()
            for j in range(len(words2) - 1):
                if words[i] + " " + words[i + 1] == words2[j] + " " + words2[j + 1]:
                    count += 1
        return count / len(words)
    def __CompareTextsTrigram(self, text: str, text2: str) -> float:
        """
            Compare trigramme d'un texte avec un autre texte

            Retourne un pourcentage de ressemblance entre les deux textes
        """
        count = 0
        words = text.split()
        for i in range(len(words) - 2):
            words2 = text2.split()
            for j in range(len(words2) - 2):
                if words[i] + " " + words[i + 1] + " " + words[i + 2] == words2[j] + " " + words2[j + 1] + " " + words2[j + 2]:
                    count += 1
        return count / len(words)


    def GetTextsMean(self) -> list[list[str, float]]:
        """
            Retourne la moyenne du texte face aux autres textes du cluster

            Retourne la moyenne de tout les textes du cluster [[text, mean], ...]]
        """
        if self.type == ClusterType.UNIGRAM:
            return self.__GetTextsMeanUnigram()
        elif self.type == ClusterType.BIGRAM:
            return self.__GetTextsMeanBigram()
        elif self.type == ClusterType.TRIGRAM:
            return self.__GetTextsMeanTrigram()
        return []
    
    def __GetTextsMeanUnigram(self) -> list[list[str, float]]:
        """
            Retourne la moyenne du texte face aux autres textes du cluster

            Retourne la moyenne de tout les textes du cluster [[text, mean], ...]]
        """
        means = []
        for text in self.texts:
            mean: float = 0
            for otherText in self.texts:
                mean += self.__CompareTextsUnigram(text, otherText)
            means.append([text, mean / len(self.texts)])
        return means
    def __GetTextsMeanBigram(self) -> list[list[str, float]]:
        """
            Retourne la moyenne du texte face aux autres textes du cluster

            Retourne la moyenne de tout les textes du cluster [[text, mean], ...]]
        """
        means = []
        for text in self.texts:
            mean: float = 0
            for otherText in self.texts:
                mean += self.__CompareTextsBigram(text, otherText)
            means.append([text, mean / len(self.texts)])
        return means
    def __GetTextsMeanTrigram(self) -> list[list[str, float]]:
        """
            Retourne la moyenne du texte face aux autres textes du cluster

            Retourne la moyenne de tout les textes du cluster [[text, mean], ...]]
        """
        means = []
        for text in self.texts:
            mean: float = 0
            for otherText in self.texts:
                mean += self.__CompareTextsTrigram(text, otherText)
            means.append([text, mean / len(self.texts)])
        return means
