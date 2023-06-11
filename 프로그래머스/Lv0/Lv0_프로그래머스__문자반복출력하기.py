#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 문자 반복 출력하기 
def solution(my_string, n):
    return ''.join(list(map(lambda x: x*n, list(my_string))))

