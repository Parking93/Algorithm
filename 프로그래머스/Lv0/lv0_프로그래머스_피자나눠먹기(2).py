# -*- coding: utf-8 -*-
"""Lv0_프로그래머스_피자나눠먹기(2).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n7V6PjgqE7MsMXFyyQRIx6jJ23iFZR4w
"""

# 피자 나눠 먹기 (2)

def solution(n):
    num_pizza = 1
    while 6*num_pizza%n != 0:
        num_pizza += 1
    return num_pizza