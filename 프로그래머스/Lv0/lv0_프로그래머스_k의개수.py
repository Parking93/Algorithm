# -*- coding: utf-8 -*-
"""Lv0_프로그래머스_k의개수.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n7V6PjgqE7MsMXFyyQRIx6jJ23iFZR4w
"""

# k의 개수
def solution(i, j, k):
    return sum([str(a).count(str(k)) for a in range(i, j+1)])