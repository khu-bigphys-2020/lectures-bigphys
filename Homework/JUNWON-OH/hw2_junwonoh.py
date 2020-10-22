#%%
# hw2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

size = 100
v = 10.0

matrix1 = np.zeros([size + 1, size + 1], float)
matrix2 = np.zeros([size + 1, size + 1], float)
matrix1[0, :] = v
matrix2[0, :] = v

for a in tqdm.tqdm(range(3000)):
    for i in range(1, size):
        for j in range(1, size):
            matrix2[i, j] = (
                matrix1[i - 1, j]
                + matrix1[i + 1, j]
                + matrix1[i, j - 1]
                + matrix1[i, j + 1]
            ) / 4

    delta = np.max(abs(matrix1 - matrix2))

    matrix1, matrix2 = matrix2, matrix1

plt.imshow(matrix1)
plt.savefig("img.png")

matrixTable = pd.DataFrame(matrix1)
matrixTable.to_csv("data1.csv")

# %%
