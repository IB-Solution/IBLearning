"""
    Word.py
    IBLearning project
    By Alix Hamidou
    2022

    Ce fichier contient la classe Word
    - text: le mot
    - count: le nombre de fois que le mot apparait dans le cluster
"""

from dataclasses import dataclass

@dataclass
class Word:
    text: str
    count: int = 0