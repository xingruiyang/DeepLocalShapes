import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import json


def load_arrays(path, filenames):
    data_array = []
    for filename in filenames:
        data_array.append(
            np.array(json.load(open(os.path.join(path, filename), 'r'))))
    return data_array


# def smooth_array(array_in):
#     array_out = array_in
#     array_len = array_in.shape[0]
#     for ind in range(1, array_len):
#         array_out[ind] = array_out[ind] + array_out[ind-1]
#     array_out /= range(1, array_len+1)
#     return array_out

def smooth_array(scalars, weight=0.5):
    array_out = np.zeros_like(scalars)
    last = scalars[0]
    for i in range(scalars.shape[0]):
        smoothed_val = last * weight + \
            (1 - weight) * scalars[i] 
        array_out[i] = smoothed_val
        last = smoothed_val

    return array_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('--smooth', action='store_true')
    args = parser.parse_args()

    config_filename = os.path.join(args.data_path, 'config.json')
    config = json.load(open(config_filename, 'r'))
    for fig in config["Figure"]:
        tags = fig["tag"]
        filenames = fig["data"]
        datas = load_arrays(args.data_path, filenames)

        for ind, tag in enumerate(tags):
            data = datas[ind]
            plt.plot(smooth_array(np.log(data[:, 2]))
                     if args.smooth else np.log(data[:, 2]), label=tag)
        plt.legend(loc="upper right")
        plt.xlabel(fig["x-axis"])
        plt.ylabel(fig["y-axis"])
        plt.suptitle(fig["title"])
        plt.show()
