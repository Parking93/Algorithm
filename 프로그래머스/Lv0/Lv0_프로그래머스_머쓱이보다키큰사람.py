# 머쓱이보다 키 큰 사람
def solution(array, height):   
    return len(list(filter(lambda x: x>height, array)))
