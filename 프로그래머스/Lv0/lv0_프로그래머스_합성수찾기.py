# -*- coding: utf-8 -*-
"""Lv0_프로그래머스_합성수찾기.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n7V6PjgqE7MsMXFyyQRIx6jJ23iFZR4w
"""

# 합성수 찾기 
def solution(n):
    answer = 0
    for i in range(1, n+1):
        count = 0
        for j in range(1,i+1):
            if i % j == 0:
                count += 1
        if count >= 3:
            answer += 1
    return answer