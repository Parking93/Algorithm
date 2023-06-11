#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 모음 제거
def solution(my_string):
    for i in ['a', 'e', 'i', 'u', 'o']:
        if i in my_string:
            my_string = my_string.replace(i, '')
    return my_string

