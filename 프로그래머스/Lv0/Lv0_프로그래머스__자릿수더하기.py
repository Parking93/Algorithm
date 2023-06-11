#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 자릿수 더하기
def solution(n):
    return(sum(list(map(lambda x: int(x), list(str(n))))))

