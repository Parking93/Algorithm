# -*- coding: utf-8 -*-


# 10번
def solution(n):
    for i in range(1, n + 1):
        print((' ' * (n-i)) + ((2 * i) - 1)*'*')

# 11번
s = 0
for i in range(1, 101):
    s += i
print(s)

# 12번
class Wizard:
    def __init__(self, health, mana, armor):
        '''
        self는 인스턴스, 인스턴스 메모리 영역
        '''
        self.health = health
        self.mana = mana
        self.armor = armor

    def attack(self):
        print('파이어볼')

jik = Wizard(health = 545, mana = 210, armor = 10)
print(jik.health, jik.mana, jik.armor)
jik.attack()

# 16번
'안녕하세요'[::-1] # 정답1

''.join(list(reversed(list('안녕하세요')))) # 정답2

s = '안녕하세요' # 정답3
ss = list(s)
ss.reverse()
''.join(ss)

s = '안녕하세요' # 정답4
result = ''
for i in s:
    result = i + result
result

def reverse(s): # 정답5
    if len(s) == 1:
        return s
    else:
        return reverse(s[1:]) + s[0]
reverse('안녕하세요')

# 17번
def solution(height): # 정답1
    if height >= 150:
        return 'YES'
    else:
        return 'NO'

def solution(height): # 정답2
    return 'YES' if height >= 150 else 'NO'

# 18번
user_input = input() # '20 30 40'
score_list = list(map(int, user_input.split(' ')))
int(sum(score_list)/len(score_list))

# 24번
def solution(s):
    return s.upper()
    
# 26번 
def solution(s):
    key = '수성, 금성, 지구, 화성, 목성, 토성, 천왕성, 해왕성'.split(', ')
    value = 'Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune'.split(', ')
    return dict(zip(key, value))[s]

# 28번 
s = 'Python' # 정답1
for i in range(len(s)-1):
    print(s[i]+s[i+1])
    
s = 'Python' # 정답2
for i in zip(s, s[1:]):
    print(''.join(i))


# 29번
# 문제 변경: 대문자만 지나가도록 코딩해주세요. 'HellO WoRld' -> HOR
for s in 'HellO WoRld':
    if s.isupper(): 
        print(s)

# 30번
def sol(s1, s2):
    return s1.find(s2)
sol('pineapple is yummy','apple')

# 35번
def one(n):
    def two(value):
        return value ** n
    return two

a = one(2)
b = one(3)
c = one(4)
print(a(10))
print(b(10))
print(c(10))

# 36번
def sol(n):
    answer = ''
    for i in range(1, 10):
        print(n*i, end= ' ')
sol(2)


# 37번
vote = '원범 원범 혜원 혜원 혜원 혜원 유진 유진'
vote_list = vote.split(' ')
vote_list.count(max(vote_list))
print(f'{max(vote_list)}(이)가 총 {vote_list.count(max(vote_list))}표로 반장이 되었습니다.')

# 38번
점수입력 = '97 86 75 66 55 97 85 97 97 95'
sorted(set(map(int, 점수입력.split(' '))), reverse=True)[2] # 3위 점수

count = 0
for i in map(int, 점수입력.split(' ')):
    if i >= sorted(set(map(int, 점수입력.split(' '))), reverse=True)[2]:
        count+=1
count

# 39번
input = input() # 정답1
print(input.replace('q','e').replace('b','n'))

import re # 정답2
def solution(s):
    return re.sub('e', 'q',re.sub('q', 'e', s))
solution('hqllo my bamq is hyqwob')

import re # 정답3 
re.sub(r'[qb]', lambda x: 'e' if x.group() == 'q' else 'n', s)

# 41번
def sol(n):
    if n == 1:
        return 'NO'
    elif n == 2:
        return 'YES'
    for i in range(2, n):
        if n % i == 0:
            return 'NO'
            break
        else:
            return 'YES'

# 43번
x = 13
answer = ''
while x:
    answer += str(x % 2)
    x //= 2
answer[::-1]

# 44번
n = 18234 # 정답1
sum(map(int, [i for i in str(n)]))

n = 18234 # 정답2
answer = 0
for i in str(n):
    answer += int(i)
answer

# 45번
import time
t = time.time()
t = int(t//(3600*24*365))+1970
print(t)




# 48번
s = 'AAABBBcccddd'
answer = ''
for i in s:
    if i.isupper():
        answer += i.lower()
    elif i.islower():
        answer += i.upper()
answer

# 49번
n = '10 9 8 7 6 5 4 3 2 1'
max(map(int,n.split(' ')))


# 53번
def sol(s):
    a = ''
    for i in s:
        if i in ('(', ')'):
            a += i
    b = ''
    for i in a:
        if i in '(':
            b += i
        elif (i in ')') and ('(' in b):
            b = b[:-1]
        else:
            return('NO')
            break
    if b == '':
        return('YES')
sol('(()))')

# 54번
def sol(num):
    num = list(map(int, num.split(' ')))
    if sorted(num) == list(range(min(num), max(num)+1)):
        return('YES')
    else:
        return('NO')
sol('1 4 2 6 3')

# 55번 
# [[A, C], [A, B] c,b a,c b,a b,c a,c]]
원판의이동경로 = []
def 하노이(원반의수, 시작기둥, 목표기둥, 보조기둥):
    #원판이 한개일 때에는 옮기면 됩니다.
    if 원반의수 == 1:
        원판의이동경로.append([시작기둥, 목표기둥])
        return None
    #원반의 n-1개를 경유기둥으로 옮기고
    하노이(원반의수 - 1, 시작기둥, 보조기둥, 목표기둥)
    #가장 큰 원반은 목표기둥으로
    원판의이동경로.append([시작기둥, 목표기둥])
    #경유기둥과 시작기둥을 바꿉니다!
    하노이(원반의수 - 1, 보조기둥, 목표기둥, 시작기둥)

하노이(3, 'A','C','B')

print(len(원판의이동경로))
원판의이동경로

# 56번
nationWidth = {      # 정답1
     'korea': 220877,
     'Rusia': 17098242,
     'China': 9596961,
     'France': 543965,
     'Japan': 377915,
     'England' : 242900 }
a = nationWidth['korea']
nationWidth.pop('korea')
gap = float('inf')
answer = ''
for nation, value in nationWidth.items():
    if abs(a - value) < gap:
        gap = abs(a - value)
        answer = nation
print(answer, value)
## 정답2
nationWidth = {       
     'Rusia': 17098242,
     'China': 9596961,
     'France': 543965,
     'Japan': 377915,
     'England' : 242900 }
def f(key):
    return [nationWidth[key] - 220877, key]
min(map(f, nationWidth))
## 정답3
nationWidth = {      
     'Rusia': 17098242,
     'China': 9596961,
     'France': 543965,
     'Japan': 377915,
     'England' : 242900 }
min(map(lambda key: [nationWidth[key] - 220877, key], nationWidth))[1]

# 57번
## 정답1
count = 0 
for i in range(0, 1001):
    count += str(i).count('1')
count
## 정답2
str(list(range(0, 1001))).count('1')

# 58번
format(123456789, ',')

# 59번
print(f'{userinput:=^50}')

# 60번
student = ['강은지','김유정','박현서','최성훈','홍유진','박지호','권윤일','김채리',
           '한지호','김진이','김민호','강채연']
for num, name in enumerate(student):
    print(f'번호: {num+1}, 이름: {name}')
    
# 61번
## 정답1
userinput = 'aaabbbcdddd'
s = userinput[0]
count = 0
answer = ''
for i in userinput:
    if i == s:
        count += 1
    else:
        answer += s
        answer += str(count)
        count = 1
        s = i
answer += s + str(count)
answer
## 정답2
import re
s = 'aaabbbbcddddaaabbbbcdddd'
answer =''
for i in re.findall('(\\w)(\\1*)', s):
    answer += i[0] + str(len(i[1])+1)
answer

# 62번
string='aacddddddddd'
a=string.count('a') #2
b=string.count('b') #0
c=string.count('c') #1
d=string.count('d') #9
print(int(str(a)+str(b)+str(c)+str(d)+str(b)+str(d)+str(a)+str(a+1)))

# 63번
## 정답1
s = '복잡한 세상 편하게 살자'
answer = ''
for i in range(len(s)):
    if i == 0:
        answer += s[i]
    elif s[i-1] == ' ':
        answer += s[i]
answer
## 정답2
s = '복잡한 세상 편하게 살자'.split(' ')
answer = ''
for i in s:
    answer += i[0]
answer

# 65번
a = [1, 2, 3, 4]
b = ['a', 'b', 'c', 'd']

answer = []
count = 0
for i, j in zip(a, b):
    count += 1
    if count % 2 == 1:
        answer.append([i, j])
    else:
        answer.append([j, i])
answer

# 66번
## 정답1
탑 = ["ABCDEF", "BCAD", "ADEFQRX", "BEDFG"]
규칙 = "ABD"
answer = []
for i in 탑:
    규칙인덱스 = [i.index(rule) for rule in 규칙 if rule in i]
    if 규칙인덱스 == sorted(규칙인덱스):
        answer.append('가능')
    else:
        answer.append('불가능')
answer

## 정답2
input = ["ABCDEF", "BCAD", "ADEFQRX", "BEDFG"]
rule = "ABD"
rule = list(rule)
result = list(map(lambda x: [x.find(r) for r in rule if x.find(r) != -1], input))
list(map(lambda x: '가능' if all(x[i] < x[i+1] for i in range(len(x)-1)) else '불가능', result))

## 정답3
string = ["ABCDEF", "BCAD", "ADEFQRX", "BEDFG"]
rule = "ABD"
def solution(s, rule):
    temp = 0
    for i in s: #i:'D', s:"ABCDEF"
        if i in rule:
            if temp > rule.index(i): # rule.index(i): 문자가 rule에 있다면 몇 번째에 있는지(1 > 3)
                return '불가능'
            temp = rule.index(i)
    return '가능'
solution(string[3], rule)


"""# 67번


"""

# 2명 - [a,b] 1번
# 3명 - [a, b], [a,c],[b,c] 3번
# 조합 (2명이 들어가는 조합)

from itertools import combinations

n = 4
nums = range(n)
combi = list(combinations(nums, 2))
combi

from itertools import combinations

n = 11
nums = range(n)
combi = list(combinations(nums, 2))
59 - len(combi)

# 68번 (풀기)






# 69번 (풀기)





"""# 70번 (풀기)

"""





"""# 문제71 : 깊이 우선 탐색

**깊이 우선 탐색**이란 목표한 노드를 찾기 위해 가장 우선순위가 높은 노드의 자식으로 깊이 들어 갔다가 목표 노드가 존재하지 않으면 처음 방문한 노드와 연결된 다른 노드부터 그 자식 노드로 파고드는 검색 방법을 말합니다.


다음과 같이 리스트 형태로 노드들의 연결 관계가 주어진다고 할 때 깊이 우선 탐색으로 이 노드들을 탐색했을 때의 순서를 공백으로 구분하여 출력하는 프로그램을 완성하세요.

1. **빨간색으로 Pass라고 되어 있는 부분을 완성**해주세요.

2. **깊이 우선 탐색을 오른쪽, 왼쪽 둘 다 구현**해보세요.

3. **리스트**로도 구현해보세요.

```
1. 데이터

graph = {'E': set(['D', 'A']),
         'F': set(['D']),
         'A': set(['E', 'C', 'B']),
         'B': set(['A']),
         'C': set(['A']),
         'D': set(['E','F'])}

2. 출력
['E', 'A', 'B', 'C', 'D', 'F']

3. 코드

graph = {
        'A': set(['B', 'C', 'E']),
        'B': set(['A']),
        'C': set(['A']),
        'D': set(['E', 'F']),
        'E': set(['A', 'D']),
        'F': set(['D'])
}

def dfs(graph, start):
    visited = []
    stack = [start]

    while stack:
        n = stack.pop()
        if n not in visited:
            visited.append(n)
            pass
    return visited

print(dfs(graph, 'E'))

```
"""

graph = {
        'A': set(['B', 'C', 'E']),
        'B': set(['A']),
        'C': set(['A']),
        'D': set(['E', 'F']),
        'E': set(['A', 'D']),
        'F': set(['D'])
}

def dfs(graph, start):
    visited = []
    stack = [start]

    while stack:
        n = stack.pop()
        if n not in visited:
            visited.append(n)
            pass
    return visited

print(dfs(graph, 'E'))

graph = {
    'A': set(['B', 'C', 'E']),
    'B': set(['A']),
    'C': set(['A']),
    'D': set(['E', 'F']),
    'E': set(['A', 'D']),
    'F': set(['D'])
}

def dfs(graph, start):
    visited = []
    stack = [start]

    while stack:
        n = stack.pop()

        if n not in visited:
            visited.append(n)
            stack.extend(graph[n])
    return visited

print(dfs(graph, 'E'))

graph = {
    'A': set(['B', 'C', 'E']),
    'B': set(['A']),
    'C': set(['A']),
    'D': set(['E', 'F']),
    'E': set(['A', 'D']),
    'F': set(['D'])
}

def dfs(graph, start):
    visited = []
    stack = [start]

    while stack:
        n = stack.pop()
        if n not in visited:
            visited.append(n)
            stack += (graph[n] - set(visited))
    return visited

print(dfs(graph, 'E'))





"""# 72번

# 73번
"""



"""# 74번

"""





# 75번



"""# 76번

전쟁이 끝난 후, A나라에서는 폐허가 된 도시를 재건하려고 합니다. 그런데 이 땅은 전쟁의 중심지였으므로 전쟁 중 매립된 지뢰가 아직도 많이 남아 있었습니다. 정부는 가장 먼저 지뢰를 제거하기 위해 수색반을 꾸렸습니다.

수색반은 가장 효율적인 지뢰 제거를 하고 싶습니다. 수색반은 도시를 격자 무늬로 나눠놓고 자신들이 수색할 수 있는 범위 내에 가장 많은 지뢰가 매립된 지역을 가장 먼저 작업하고 싶습니다.

가장 먼저 테스트 케이스의 수를 나타내는 1이상 100 이하의 자연수가 주어집니다.
각 테스트 케이스의 첫 줄에는 수색할 도시의 크기 a와 수색반이 한번에 수색 가능한 범위 b가 주어집니다. (a와 b 모두 정사각형의 가로 또는 세로를 나타냅니다. 예를들어 10이 주어지면 10x10칸의 크기를 나타냅니다.)

그 후 a줄에 걸쳐 도시 내 지뢰가 있는지의 여부가 나타납니다.
0은 지뢰가 없음 1은 지뢰가 있음을 뜻합니다.

각 테스트 케이스에 대해 수색 가능한 범위 bxb 내에서 찾아낼 수 있는 가장 큰 지뢰의 갯수를 구하세요.

```
입력
1
5 3
1 0 0 1 0
0 1 0 0 1
0 0 0 1 0
0 0 0 0 0
0 0 1 0 0

출력
3

```
"""

사각형 = 5
탐색가능지역 = 3

지뢰밭 = [[1, 0, 0, 1, 0],
          [0, 1, 0, 0, 1],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0]]

import numpy as np

arr = np.array(지뢰밭)

arr[0:3, 0:3].sum

arr[0:3, 0:3].sum()

arr

print(arr[0:3, 0:3])
print(arr[0:3, 1:4])
print(arr[0:3, 2:5])

print(arr[1:4, 0:3])
print(arr[1:4, 1:4])
print(arr[1:4, 2:5])

사각형 = 5
탐색가능지역 = 3

지뢰밭 = [[1, 0, 0, 1, 0],
          [0, 1, 0, 0, 1],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0]]
arr = np.array(지뢰밭)
answer = []
for i in range(사각형 - 탐색가능지역 + 1):
    for j in range(사각형 - 탐색가능지역 + 1):
        answer.append(arr[i:(i+탐색가능지역), j:(j+탐색가능지역)].sum())

max(answer)



# 라이브러리 사용금지

answer = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]

# sum(answer) # error
sum(answer, [])
sum(sum(answer, []))



"""# 문제77 : 가장 긴 공통 부분 문자열

가장 긴 공통 부분 문자열(Longest Common Subsequence)이란 A, B 두 문자열이 주어졌을 때
두 열에 공통으로 들어 있는 요소로 만들 수 있는 가장 긴 부분열을 말합니다.
여기서 부분열이란 다른 문자열에서 몇몇의 문자가 빠져 있어도 순서가 바뀌지 않은 열을 말합니다.

예를 들어 S1 = ['T', 'H', 'E', 'S', 'T', 'R', 'I', 'N', 'G', 'S']  S2 = ['T', 'H', 'I', 'S', 'I', 'S']라는 두 문자열이 있을 때
둘 사이의 부분 공통 문자열의 길이는 ['T', 'H', 'S', 'T', 'I', 'S'] 의 6개가 됩니다.

이처럼 두 문자열이 주어지면 가장 긴 부분 공통 문자열의 길이를 반환하는 프로그램을 만들어 주세요.

두 개의 문자열이 한 줄에 하나씩 주어집니다.
문자열은 알파벳 대문자로만 구성되며 그 길이는 100글자가 넘어가지 않습니다.

출력은 이 두 문자열의 가장 긴 부분 공통 문자열의 길이를 반환하면 됩니다.
```
- Test Case -

입력
THISISSTRINGS
THISIS

출력
6

-

입력
THISISSTRINGS
TATHISISKKQQAEW

출력
6

-

입력
THISISSTRINGS
KIOTHIKESSISKKQQAEW

출력
3

-

입력
THISISSTRINGS
TKHKIKSIS

출력
3
```
"""



"""# 문제78 : 원형테이블"""

from collections import deque


n = 6
f = 2
food = list(range(1, n+1))
food

food_r = deque(food)
food_r.popleft()
food_r.rotate(-2)
food_r

food_r = deque(food)
print(f'먹은 음식: {food_r.popleft()}')
food_r.rotate(-2)
print(f'회전 후: {food_r}')

print(f'먹은 음식: {food_r.popleft()}')
food_r.rotate(-2)
print(f'회전 후: {food_r}')

print(f'먹은 음식: {food_r.popleft()}')
food_r.rotate(-2)
print(f'회전 후: {food_r}')

from collections import deque
def solution(n, k):
    food = range(1, n+1)
    food_r = deque(food)

    while len(food_r) > 2:
        food_r.popleft()
        food_r.rotate(-(k-1))
    return list(food_r)
solution(6, 3)



"""# 문제79 : 순회하는 리스트"""





"""# 문제80 : 순열과 조합"""



"""# 문제81 : 지뢰찾기

```
데이터
flag = [] #지뢰 없이 깃발만 있는 리스트
minesweeper = [] #지뢰를 찾은 리스트
for i in range(5):
    flag.append(input('깃발 값과 함께 입력하세요 :').split(' '))

pass
print(flag)
print(minesweeper)



입력
0 1 0 0 0
0 0 0 0 0
0 0 0 1 0
0 0 1 0 0
0 0 0 0 0

출력
* f * 0 0
0 * 0 * 0
0 0 * f *
0 * f * 0
0 0 * 0 0

```
"""

import numpy as np

matrix = [
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0]
]

matrix_arr = np.array(matrix)
matrix_arr = np.where(matrix_arr==1, 'f', matrix_arr)

for v, h in zip(np.where(matrix_arr=='f')[0], np.where(matrix_arr=='f')[1]):
    if v != 0:
        matrix_arr[v-1, h] = '*'
    if h != 0:
        matrix_arr[v, h-1] = '*'
    if v+1 < matrix_arr.shape[0]:
        matrix_arr[v+1, h] = '*'
    if h+1 < matrix_arr.shape[1]:
        matrix_arr[v, h+1] = '*'

print(matrix_arr)





"""# 84번

"""

from itertools import permutations
n = 1732
n_list = [int(i) for i in str(n)]
p = list(permutations(n_list, 2))
max_num = max(p)
answer = int(''.join(map(str, max_num)))
answer

# 민규님 답
number = int(input('숫자 입력: '))
k = int(input('k값 입력: '))

number_str = str(number)
number_str = sorted(number_str, reverse=True)
large_number = int(''.join(map(str, number_str[:k])))

print(large_number)

"""# 85번"""

def evalue(value):
    return ''.join([str(i) + str(value.count(str(i))) for i in range(1, int(max(value)) + 1)])

def solution(n):
    answer = '1'
    if n == 1:
        return 1
    else:
        for i in range(1, n):
            answer = evalue(answer)
            print(answer)
    return answer

# evalue('1121')
solution(6)

"""# 86번"""

from collections import deque
point = [1,1,3,2,5]
dish = 3
count = 0
point_r = deque(point)

while dish != point_r[0]:

    if dish > point_r[0]:
        point_r.popleft()
    else:
        point_r.rotate(-1)
    count += 1
count

point = [1,1,3,2,5]
dish = 3
count = 0
point_r = deque(point)
point_r.popleft()
point_r

from collections import deque

point = [1, 1, 3, 2, 5]
dish = 3
count = 0
point_r = deque(point)

while dish not in point_r:
    if dish > point_r[0]:
        point_r.popleft()
    else:
        point_r.rotate(-1)
    count += 1

print(count)

from collections import deque

point = [5,2,3,1,2,5]
dish = 1
count = 0
point_r = deque(point)

while True:
    if dish > point_r[0]:
        point_r.popleft()
    elif (dish == point_r[0]) & (dish == min(point_r)):
        break
    point_r.rotate(-1)
    count += 1
count





