import flwr 
import torch
import numpy as np
import random
import copy
# from Network.pytorch3dunet.unet3d.losses import BCEDiceLoss
from flwr.common import (
    Code,
    Context,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class CustomNumpyClient(flwr.client.NumPyClient):
    context: Context
    def __init__(self, net, train_loader, epoch, lossf, optimizer, DEVICE, args, trainF=lambda x: x, validF=lambda x: x):
        super().__init__()
        self.net = net
        self.train_loader = train_loader
        self.epoch = epoch
        self.lossf = lossf
        self.optim = optimizer
        self.DEVICE=DEVICE
        self.train = trainF
        self.valid = validF
        self.args = args
    def set_parameters(self, parameters):
        for old, new in zip(self.net.parameters(), parameters):
            old.data = torch.Tensor(new).to(self.DEVICE)
    def get_parameters(self, config={}):
        return [val.cpu().detach().numpy() for val in self.net.parameters()]
    def fit(self, parameters, config={}):
        self.set_parameters(parameters)
        def proxy_lossf(outputs, targets):
            return self.lossf(outputs, targets)+(config["tau"] / 2) \
                * sum([np.linalg.norm(n.flatten().cpu().detach().numpy()-g.flatten(), 2) for n, g in zip(self.net.parameters(), copy.deepcopy(parameters))])
        self.train(self.net, self.train_loader, None, self.epoch, proxy_lossf, self.optim, self.DEVICE, None)
        history = self.valid(self.net, self.valid_loader, 0, self.lossf, self.DEVICE, True)
        return self.get_parameters(config={}), len(self.train_loader), history

def seeding(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)