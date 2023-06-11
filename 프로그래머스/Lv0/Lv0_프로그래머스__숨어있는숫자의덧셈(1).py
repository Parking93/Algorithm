#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 숨어있는 숫자의 덧셈 (1)
import re
def solution(my_string):
    return sum(list(map(lambda x: int(x), re.findall(r'(\d)', my_string))))

