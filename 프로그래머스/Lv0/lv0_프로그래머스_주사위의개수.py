# -*- coding: utf-8 -*-
"""Lv0_프로그래머스_주사위의개수.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n7V6PjgqE7MsMXFyyQRIx6jJ23iFZR4w
"""

# 주사위의개수
def solution(box, n):
    return (box[0]//n) * (box[1]//n) * (box[2]//n)