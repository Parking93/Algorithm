# 길이에 따른 연산
def solution(num_list):
    answer_mul = 1
    if len(num_list) <= 10:
        for i in num_list:
            answer_mul *= i
        return answer_mul
    return sum(num_list)

