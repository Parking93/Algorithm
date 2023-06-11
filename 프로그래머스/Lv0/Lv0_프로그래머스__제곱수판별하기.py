#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 제곱수 판별하기
def solution(n):
    if n in list(map(lambda x: x**2, range(1, 1001))):
        return 1
    return 2

