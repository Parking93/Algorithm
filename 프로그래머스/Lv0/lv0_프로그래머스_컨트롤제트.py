# -*- coding: utf-8 -*-
"""Lv0_프로그래머스_컨트롤제트.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n7V6PjgqE7MsMXFyyQRIx6jJ23iFZR4w
"""

def solution(s):
    answer = []
    for i in s.split(' '):
        if i == 'Z':
            if len(answer) > 0:
                answer.pop()
        else:
            answer.append(int(i))
    return sum(answer)