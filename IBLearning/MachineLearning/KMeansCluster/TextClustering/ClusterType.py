"""
    ClusterType.py
    IBLearning project
    By Alix Hamidou
    2022

    Ce fichier contient les types de cluster
    - Unigram: 1 mot
    - Bigram: 2 mots
    - Trigram: 3 mots
"""

from dataclasses import dataclass

@dataclass
class ClusterType:
    UNIGRAM: int = 1           # 1 mots
    BIGRAM: int = 2             # 2 mots
    TRIGRAM: int = 3            # 3 mots
    