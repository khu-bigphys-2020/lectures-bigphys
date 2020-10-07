import random
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

radian = 3


# y = k*x 의 확률함수를 가진 -radian ~ radian 사이에 있는 입자
def norm_f(x):
    if x >= 0:
        x = sqrt(x)
    else:
        x = -(sqrt(abs(x)))
    x *= radian
    return x

nExp = 100000
nTrial = 0

x_list, y_list, z_list = [], [], []
while(True):
    #  -radian ~ radian 안에 있는 임의의 입자에 x, y, z의 좌표 설정
    x, y, z = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
    x = norm_f(x)
    y = norm_f(y)
    z = norm_f(z)

    # 그 입자가 반지름 1인 구 안에 있을 때만 입자 추가
    if (x**2 + y**2 + z**2) <= (radian**2):
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    # 구 안에 있는 입자 수가 nExp 보다 커지면 종료
        nTrial += 1
        if nTrial >= nExp:
            break

print(len(x_list), len(y_list), len(z_list))

cordinate_particle = [10, 0, 0]

force_vector_xlist = []
force_vector_ylist = []
force_vector_zlist = []

#force vector 구하기
for i in range(len(x_list)):
    distance = sqrt((x_list[i]-10)**2 + y_list[i]**2 + z_list[i]**2)
    unit_vector = [(10 - x_list[i])/distance, -y_list[i]/distance, -z_list[i]/distance]
    force_scalar = 10 * 10 / distance**2
    force_vector_x = unit_vector[0] * force_scalar
    force_vector_y = unit_vector[1] * force_scalar
    force_vector_z = unit_vector[2] * force_scalar
    force_vector_xlist.append(force_vector_x)
    force_vector_ylist.append(force_vector_y)
    force_vector_zlist.append(force_vector_z)


plt.hist(force_vector_xlist, range=(0, 2), bins=1000)
plt.show()
plt.hist(force_vector_ylist, range=(-0.5, 0.5), bins=1000)
plt.show()
plt.hist(force_vector_zlist, range=(-0.5, 0.5), bins=1000)
plt.show()

force_vector_mean = [0, 0, 0]
for i in range(len(force_vector_xlist)):
    force_vector_mean[0] += force_vector_xlist[i]
    force_vector_mean[1] += force_vector_ylist[i]
    force_vector_mean[2] += force_vector_zlist[i]

for i in range(3):
    force_vector_mean[i] /= len(force_vector_xlist)

print(force_vector_mean)
