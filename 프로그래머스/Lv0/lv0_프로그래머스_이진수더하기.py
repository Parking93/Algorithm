# -*- coding: utf-8 -*-
"""Lv0_프로그래머스_이진수더하기.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n7V6PjgqE7MsMXFyyQRIx6jJ23iFZR4w
"""

# 이진수 더하기
def solution(bin1, bin2):
    return bin(int(bin1, 2) + int(bin2, 2))[2:]