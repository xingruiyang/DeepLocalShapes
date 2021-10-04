import torch
import os


def load_model(file_name, network, device=torch.device('cpu')):
    if not os.path.isfile(file_name):
        raise Exception(
            'model state dict "{}" does not exist'.format(file_name))

    data = torch.load(file_name, map_location=device)
    network.load_state_dict(data["model_state_dict"])
    return data["epoch"]
