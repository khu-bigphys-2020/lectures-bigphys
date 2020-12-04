#%%
import random
from math import *
import matplotlib.pyplot as plt

# 구 설정
R = int(input("구의 반지름 : "))
nExp = int(input("입자의 갯수 : "))
x_list, y_list, z_list = [], [], []
for i in range(nExp):
    r = random.uniform(0, R)
    r = r ** (1 / 2)
    theta = random.uniform(0, 180)
    phi = random.uniform(0, 360)
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)

plt.plot(x_list, y_list, ".")
plt.show()
plt.plot(x_list, z_list, ".")
plt.show()

# 임의의 좌표 설정
location = []
while True:
    x = random.uniform(0, 3 * R)
    y = random.uniform(0, 3 * R)
    z = random.uniform(0, 3 * R)
    if x ** 2 + y ** 2 + z ** 2 < R ** 2:
        location.append(x)
        location.append(y)
        location.append(z)
        break
    else:
        continue
print(f"임의의 입자의 좌표 : {location}")

# Electric Field 계산
force = []
force_x, force_y, force_z = [], [], []
for i in range(nExp):
    dif_x = location[0] - x_list[i]
    dif_y = location[1] - y_list[i]
    dif_z = location[2] - z_list[i]
    l = sqrt(dif_x ** 2 + dif_y ** 2 + dif_z ** 2)
    f = 1 / (l ** 2)
    force.append(f)
    force_x.append(dif_x * f)
    force_y.append(dif_y * f)
    force_z.append(dif_z * f)
print(f"힘의 합 : {sum(force)}")
print(f"힘의 평균 : {sum(force)/nExp}")
print(f"x축 벡터 평균 : {sum(force_x) / len(force_x)}")
print(f"y축 벡터 평균 : {sum(force_y) / len(force_y)}")
print(f"z축 벡터 평균 : {sum(force_z) / len(force_z)}")
# %%
