import matplotlib.pyplot as plt
import numpy as np


file1 = 'benchmark/chamfer_train_aligned.txt'
file2 = 'benchmark/chamfer_train_unaligned.txt'

a = np.loadtxt(file1)
b = np.loadtxt(file2)

# print(a)

plt.plot(a[:, 2], label='aligned')
plt.plot(b[:, 2], label='unaligned')
plt.xlabel("num epochs")
plt.ylabel("chamfer distance")
plt.legend(loc="upper right")
plt.show()