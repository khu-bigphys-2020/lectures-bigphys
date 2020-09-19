#%%
import matplotlib.pyplot as plt

# 초기 조건
m = 1  # 입자의 질량
t = 0  # 시간
k = 0.1  # 용수철 상수
x_0 = 1.0  # 용수철 길이

# 입자의 초기 변위
dx_1, dx_2, dx_3, dx_4 = 0, 0, 0, 0.5
# 입자의 초기 위치
x_1, x_2, x_3, x_4 = x_0 + dx_1, 2 * x_0 + dx_2, 3 * x_0 + dx_3, 4 * x_0 + dx_4
# 입자의 초기 속도
v_1, v_2, v_3, v_4 = 0, 0, 0, 0

dt = 0.1

t_list = []
x_list = [[], [], [], []]
v_list = [[], [], [], []]

for i in range(1000):
    t += dt
    # 용수철 힘 (F = -kx)
    f_1 = -k * (x_1 - x_0)
    f_2 = -k * (x_2 - x_1 - x_0)
    f_3 = -k * (x_3 - x_2 - x_0)
    f_4 = -k * (x_4 - x_3 - x_0)
    # 입자의 가속도 (F = ma = -kx)
    a_1 = (f_1 - f_2) / m
    a_2 = (f_2 - f_3) / m
    a_3 = (f_3 - f_4) / m
    a_4 = f_4 / m
    # 입자의 속도
    v_1 += a_1 * dt
    v_2 += a_2 * dt
    v_3 += a_3 * dt
    v_4 += a_4 * dt
    # 입자의 위치
    x_1 += v_1 * dt
    x_2 += v_2 * dt
    x_3 += v_3 * dt
    x_4 += v_4 * dt

    t_list.append(t)
    x_list[0].append(x_1)
    x_list[1].append(x_2)
    x_list[2].append(x_3)
    x_list[3].append(x_4)
    v_list[0].append(v_1)
    v_list[1].append(v_2)
    v_list[2].append(v_3)
    v_list[3].append(v_4)
# X-T 그래프
plt.plot(t_list, x_list[0], label="1")
plt.plot(t_list, x_list[1], label="2")
plt.plot(t_list, x_list[2], label="3")
plt.plot(t_list, x_list[3], label="4")
plt.xlabel("T")
plt.ylabel("X")
plt.title("X-T Graph")
plt.show()

# X-V 그래프
for i in range(4):
    plt.plot(v_list[i], x_list[i])
    plt.title(f"{i} X-T")
    plt.xlabel("V")
    plt.ylabel("X")
    plt.show()

# %%
