# 옷가게 할인 받기
def solution(price):
    if price >= 500000:
        return int(price*0.80)
    elif price >= 300000:
        return int(price*0.90)
    elif price >= 100000:
        return int(price*0.95)
    return int(price)
