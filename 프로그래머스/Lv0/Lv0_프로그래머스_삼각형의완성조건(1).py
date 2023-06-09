# 삼각형의 완성조건 (1)
def solution(sides):
    a = sorted(sides, reverse=True)
    if a[0] > a[1]+a[2]:
        return 2
    return 1