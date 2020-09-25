#%%
import matplotlib.pyplot as plt

# 초기 조건
m = 1  # 입자의 질량
t = 0  # 시간
k = 0.1  # 용수철 상수
x_0 = 1.0  # 용수철 길이
dt = 0.1
# 입자 갯수 설정
n = int(input("입자 갯수 : "))
# 초기 변위 설정
dx_list = []
for i in range(n):
    dx = float(input(f"{i + 1}번째 입자의 초기 변위 : "))
    dx_list.append(dx)
# 시간, 속도, 변위, 가속도, 힘 리스트 설정
t_list = [0.1]
x_list = []
v_list = []
a_list = []
f_list = []
for i in range(n):
    x_list.append([])
    v_list.append([])
    a_list.append([])
    f_list.append([])
    # 초기 위치 설정
    x_list[i].append(x_0 * (i + 1) + dx_list[i])
    v_list[i].append(0)
print(x_list)


for i in range(1000):
    t += dt
    t_list.append(t)
    f_list[0].append(-k * (x_list[0][-1] - x_0))
    for j in range(1, n):
        f_list[j].append(-k * (x_list[j][-1] - x_list[j - 1][-1] - x_0))
    for k in range(n - 1):
        a_list[k].append((f_list[k][-1] - f_list[k + 1][-1]) / m)
    a_list[-1].append(f_list[-1][-1] / m)
    for l in range(n):
        v_list[l].append(v_list[l][-1] + a_list[l][-1] * dt)
    for m in range(n):
        x_list[m].append(x_list[m][-1] + v_list[l][-1] * dt)
for i in range(n):
    plt.plot(t_list, x_list[i])
plt.show()

# %%
