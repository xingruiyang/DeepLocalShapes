from sklearn.neighbors import NearestNeighbors
import os

import numpy as np
import torch


def save_model(folder, filename, model, epoch=0):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
               os.path.join(folder, filename))


def load_model_parameters(file_name, decoder, device=torch.device('cpu')):
    if not os.path.isfile(file_name):
        raise Exception(
            'model state dict "{}" does not exist'.format(file_name))

    data = torch.load(file_name, map_location=device)
    decoder.load_state_dict(data["model_state_dict"])
    return data["epoch"]


def save_optimizer(folder, filename, optimizer, epoch=0):
    torch.save({"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
               os.path.join(folder, filename),)


def load_optimizer(file_name, optimizer):
    data = torch.load(file_name)
    epoch = data['epoch']
    optimizer.load_state_dict(data['optimizer_state_dict'])
    return epoch


def save_latents(folder, filename, latents):
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    np.save(os.path.join(folder, filename), latents)


def load_latents(filename, device=torch.device('cpu')):
    latents = np.load(filename)
    latents = torch.from_numpy(latents).float().to(device)
    return latents


def save_latest(folder, network, optimizer, latents=None, epoch=0):
    print("The latest model is saved to {}".format(folder))
    save_model(
        folder, "latest_model.pth", network, epoch)
    save_optimizer(
        folder, "latest_optim.pth", optimizer, epoch)
    if latents is not None:
        save_latents(folder, "latest_latents.npy", latents)


def save_ckpts(folder, decoder, optimizer, latents=None, shape=None, epoch=0):
    print("Check point file is saved to {}".format(folder))
    base_name = "ckpt_" + str(epoch) + "_"
    save_model(folder, base_name + "model.pth", decoder, epoch)
    save_optimizer(folder, base_name + "optim.pth", optimizer, epoch)
    if latents is not None:
        save_latents(folder, base_name + "latents.npy", latents)
    if shape is not None:
        shape.export(os.path.join(folder, base_name+"mesh.ply"))


def load_model(file_name, decoder, device=torch.device('cpu')):
    if not os.path.isfile(file_name):
        raise Exception(
            'model state dict "{}" does not exist'.format(file_name))

    data = torch.load(file_name, map_location=device)
    decoder.load_state_dict(data["model_state_dict"])
    return data["epoch"]


def log_progress(step, total, prefix="", suffix=""):
    fill = '█'
    bar_length = 30
    percent = ("{0:.2f}").format(100 * (step / float(total)))
    filled = int(bar_length * step // total)
    bar = fill * filled + '-' * (bar_length - filled)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
    if step == total:
        print()


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError(
            "Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist
