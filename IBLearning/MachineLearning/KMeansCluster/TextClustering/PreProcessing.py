"""
    PreProcessing.py
    IBLearning project
    By: Alix Hamidou
    2022

    Permet de prétraiter les données textuelles
    - Retirer les stop words
    - Retirer la ponctuation
    - Mettre en minuscules

    Exemple d'utilisation:
    >>> from PreProcessing import Preprocess
    >>> text = "Ceci est un texte de test. Il contient des mots inutiles."
    >>> print(Preprocess(text))
    ceci texte test contient mots inutiles
"""

import string
from dataclasses import dataclass

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


@dataclass
class Language:
    """
        Langue pour la prétraitement des données textuelles
    """
    catalan: str = "catalan"
    czech: str = "czech"
    german: str = "german"
    greek: str = "greek"
    english: str = "english"
    spanish: str = "spanish"
    finnish: str = "finnish"
    french: str = "french"
    hungarian: str = "hungarian"
    icelandic: str = "icelandic"
    italian: str = "italian"
    latvian: str = "latvian"
    dutch: str = "dutch"
    polish: str = "polish"
    portuguese: str = "portuguese"
    romanian: str = "romanian"
    russian: str = "russian"
    slovak: str = "slovak"
    slovenian: str = "slovenian"
    swedish: str = "swedish"
    tamil: str = "tamil"


def Preprocess(text: str, language: str) -> str:
    # convertir en minuscules
    text = text.lower()
    
    # suppression de la ponctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # suppression des stopwords
    stop_words = set(stopwords.words(language))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # reconstruire le texte prétraité
    text = " ".join(words)
    return text