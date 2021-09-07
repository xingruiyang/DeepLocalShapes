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


def save_latest(folder, decoder, optimizer, latents=None, epoch=0):
    print("The latest model is saved to {}".format(folder))
    save_model(
        folder, "latest_model.pth", decoder, epoch)
    save_optimizer(
        folder, "latest_optim.pth", optimizer, epoch)
    if latents is not None:
        save_latents(folder, "latest_latents.npy", latents)


def save_checkpoints(folder, decoder, optimizer, latents=None, epoch=0):
    print("Check point file is saved to {}".format(folder))
    base_name = "ckpt_epoch_" + str(epoch) + "_"
    save_model(folder, base_name + "model.pth", decoder, epoch)
    save_optimizer(folder, base_name + "optim.pth", optimizer, epoch)
    if latents is not None:
        save_latents(folder, base_name + "latents.npy", latents)


def load_model_parameters(file_name, decoder, device=torch.device('cpu')):
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
