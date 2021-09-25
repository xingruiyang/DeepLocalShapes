import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('file1')
    parser.add_argument('file2')
    args = parser.parse_args()

    # file1 = 'output//eval_aligned.txt'
    # file2 = 'benchmark/eval_unaligned.txt'

    a = np.loadtxt(args.file1)
    b = np.loadtxt(args.file2)

    # print(a)

    plt.plot(a[:, 1], label='aligned')
    plt.plot(b[:, 1], label='unaligned')
    plt.xlabel("num epochs")
    plt.ylabel("chamfer distance")
    plt.legend(loc="upper right")
    plt.savefig("out.png")