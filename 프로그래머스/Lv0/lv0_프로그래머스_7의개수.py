# -*- coding: utf-8 -*-
"""Lv0_프로그래머스_7의개수.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n7V6PjgqE7MsMXFyyQRIx6jJ23iFZR4w
"""

# 7의 개수
def solution(array):
    array_str = [str(num) for num in array]
    return ''.join(array_str).count('7')

