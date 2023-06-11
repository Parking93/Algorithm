#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 가위 바위 보

def solution(rsp):
    rsp_list = list(rsp)
    answer = []
    for i in rsp_list:
        if i == '2':
            answer.append('0')
        elif i == '0':
            answer.append('5')
        else:
            answer.append('2')
    return ''.join(answer)

