import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

grid = 100
voltage = 10.0

phi = np.zeros([grid + 1, grid + 1], float)
phitmp = np.zeros([grid + 1, grid + 1], float)
phi[0, :] = voltage
phitmp[0, :] = voltage

for a in tqdm.tqdm(range(3000)):
    for i in range(1, grid):
        for j in range(1, grid):
            phitmp[i, j] = (
                phi[i - 1, j] + phi[i + 1, j] + phi[i, j - 1] + phi[i, j + 1]
            ) / 4

    delta = np.max(abs(phi - phitmp))

    phi, phitmp = phitmp, phi

plt.imshow(phi)
plt.savefig("Plot by plt.imshow().png")

phiTable = pd.DataFrame(phi)
phiTable.to_csv("DataSet.csv")