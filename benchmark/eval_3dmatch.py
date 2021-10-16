import math
import argparse

import numpy as np
from itertools import chain


def read_index_and_pose(filename):
    results_ind = []
    results_pose = []
    with open(filename, "r") as f:
        lines = f.readlines()

    num_items = len(lines) // 5
    for ind in range(num_items):
        indices = np.array(lines[ind*5].strip().split(' ')).astype(int)
        pose = np.eye(4)
        pose[0, :] = np.array(lines[ind*5+1].strip().split()).astype(float)
        pose[1, :] = np.array(lines[ind*5+2].strip().split()).astype(float)
        pose[2, :] = np.array(lines[ind*5+3].strip().split()).astype(float)
        pose[3, :] = np.array(lines[ind*5+4].strip().split()).astype(float)

        results_ind.append(indices)
        results_pose.append(pose)

    # results_ind = np.stack(results_ind)
    # results_pose = np.stack(results_pose)
    return results_ind, results_pose


def read_information_matrix(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    num_items = len(lines) // 7
    gt_info = []
    for ind in range(num_items):
        info = np.eye(6)
        info[0, :] = np.array(lines[ind*7+1].strip().split()).astype(float)
        info[1, :] = np.array(lines[ind*7+2].strip().split()).astype(float)
        info[2, :] = np.array(lines[ind*7+3].strip().split()).astype(float)
        info[3, :] = np.array(lines[ind*7+4].strip().split()).astype(float)
        info[4, :] = np.array(lines[ind*7+5].strip().split()).astype(float)
        info[5, :] = np.array(lines[ind*7+6].strip().split()).astype(float)

        gt_info.append(info)
    return gt_info


def dcm2quat(DCM):
    qout = np.zeros((4,))
    qout[0] = 0.5 * math.sqrt(1 + DCM[0, 0] + DCM[1, 1] + DCM[2, 2])
    qout[1] = -(DCM[2, 1] - DCM[1, 2]) / (4 * qout[0])
    qout[2] = -(DCM[0, 2] - DCM[2, 0]) / (4 * qout[0])
    qout[3] = -(DCM[1, 0] - DCM[0, 1]) / (4 * qout[0])
    return qout


def compute_transformation_error(trans, info):
    te = trans[:3, 3]
    qt = dcm2quat(trans[:3, :3])
    er = np.concatenate([te, -qt[1:4]])
    p = np.matmul(er.reshape(1, 6), info)
    p = np.matmul(p, er.reshape(6, 1))
    p /= info[0, 0]
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=str)
    parser.add_argument('gt', type=str)
    parser.add_argument('gt_info', type=str)
    parser.add_argument('--err2', type=float, default=0.04)
    args = parser.parse_args()

    results_ind, results_pose = read_index_and_pose(args.result)
    gt_ind, gt_pose = read_index_and_pose(args.gt)
    gt_info = read_information_matrix(args.gt_info)
    num_fragments = gt_ind[0][-1]

    mask = np.zeros((num_fragments, num_fragments), dtype=int)
    gt_num = 0
    for i in range(len(gt_ind)):
        if gt_ind[i][1] - gt_ind[i][0] > 1:
            mask[gt_ind[i][0],  gt_ind[i][1]] = i
            gt_num += 1

    rs_num = 0
    good = 0
    for i in range(len(results_ind)):
        if results_ind[i][1] - results_ind[i][0] > 1:
            rs_num += 1
            idx = mask[results_ind[i][0],  results_ind[i][1]]
            if idx != 0:
                rel_pose = np.matmul(
                    gt_pose[idx], np.linalg.inv(results_pose[i]))
                p = compute_transformation_error(rel_pose, gt_info[idx])
                if p <= args.err2:
                    good += 1
                # else:
                #     print(results_ind[i])

    print(gt_num, rs_num)
    recall = good / gt_num
    precision = good / rs_num
    print("precision: {} recall {}".format(precision, recall))
