import inspect
import os

import torch
import torch.nn as nn

activations = {
    "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
    "soft_plus": nn.Softplus(beta=100)
}

optimizers = {
    "Adam": torch.optim.Adam
}


class ImplicitNet(nn.Module):
    def __init__(self,
                 latent_dim,
                 hidden_dims,
                 use_tanh=False,
                 voxel_size=1,
                 clamp_dist=-1,
                 act_fn="leaky_relu",
                 optimizer="Adam"):
        super(ImplicitNet, self).__init__()

        dims = [latent_dim+3] + hidden_dims + [1]
        self.num_layers = len(dims)

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)
            setattr(self, "lin_"+str(layer), lin)

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()

        self.act_fn = activations.get(
            act_fn, nn.ReLU())
        self.optimizer = optimizers.get(
            optimizer, torch.optim.Adam)

        self.clamp_dist = clamp_dist
        self.input_scale = 1.0/voxel_size
        self.latent_vecs = None
        self.latent_dim = latent_dim

        self.hparams = self.get_hparams()

    def forward(self, inputs):
        x = inputs
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin_" + str(layer))
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.act_fn(x)
        return self.tanh(x) if self.use_tanh else x

    def save_model(self, filename, epoch=0):
        model_state_dict = {
            "epoch": epoch,
            "hparams": self.hparams,
            "model_state_dict": self.state_dict()}
        torch.save(model_state_dict, filename)

    @staticmethod
    def load_from_ckpt(cls, filename, device=torch.device('cpu')):
        if not os.path.isfile(filename):
            raise Exception(
                'model state dict "{}" does not exist'.format(filename))

        state_dict = torch.load(filename, map_location=device)
        network = ImplicitNet(**state_dict['hparams']).to(device)
        network.load_state_dict(state_dict["model_state_dict"])
        return network, state_dict["epochs"]

    def get_hparams(self):
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        cls = local_vars["__class__"]
        init_parameters = inspect.signature(cls.__init__).parameters
        init_params = list(init_parameters.values())
        n_self = init_params[0].name

        def _get_first_if_any(params, param_type):
            for p in params:
                if p.kind == param_type:
                    return p.name
            return None

        n_args = _get_first_if_any(
            init_params, inspect.Parameter.VAR_POSITIONAL)
        n_kwargs = _get_first_if_any(
            init_params, inspect.Parameter.VAR_KEYWORD)
        filtered_vars = [n for n in (n_self, n_args, n_kwargs) if n]
        exclude_argnames = (*filtered_vars, "__class__", "frame", "frame_args")
        # only collect variables that appear in the signature
        local_args = {k: local_vars[k] for k in init_parameters.keys()}
        # kwargs_var might be None => raised an error by mypy
        if n_kwargs:
            local_args.update(local_args.get(n_kwargs, {}))
        local_args = {k: v for k, v in local_args.items()
                      if k not in exclude_argnames}
        return local_args
