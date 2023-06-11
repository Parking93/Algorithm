#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 짝수는 싫어요
def solution(n):
    return list(filter(lambda x: x%2!=0, range(1,n+1)))

