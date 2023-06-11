#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 개미군단

def solution(hp):
    count = 0 
    while hp > 0:
        if hp >= 5:
            hp -= 5
            count += 1
        elif hp >= 3:
            hp -= 3
            count += 1
        elif hp >= 1:
            hp -= 1
            count += 1
    return count 

