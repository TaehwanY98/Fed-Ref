import flwr 
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from train import train, valid, CustomFocalDiceLoss
import warnings
from utils.parser import Federatedparser
from utils.CustomDataset import Fets2022
import numpy as np
import random
from Network.Unet import *
from Network.Loss import *
import getpass
import sys
user = getpass.getuser()
sys.path.append(f"/home/{user}/MICCAI/Network")
# from Network.pytorch3dunet.unet3d.losses import BCEDiceLoss
from flwr.common import (
    Code,
    Context,
    EvaluateIns,
    EvaluateRes,
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Federatedparser()
net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
net.to(DEVICE)


class CustomClient(flwr.client.Client):
    context: Context
    def __init__(self, net, train_loader, epoch, lossf, optimizer, DEVICE, args,trainF=train, validF=valid):
        super().__init__()
        self.net = net
        self.train_loader = train_loader
        self.epoch = epoch
        self.lossf = lossf
        self.optim = optimizer
        self.DEVICE=DEVICE
        # dataset = Fets2022(args.data_dir)
        # _, validset = random_split(dataset, [0.9, 0.1])
        # self.valid_loader = DataLoader(validset, args.batch_size, False, collate_fn=lambda x: x)
        self.train = trainF
        self.valid = validF
    
    """Abstract base class for Flower clients."""
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return the current local model parameters.

        Parameters
        ----------
        ins : GetParametersIns
            The get parameters instructions received from the server containing
            a dictionary of configuration values.

        Returns
        -------
        GetParametersRes
            The current local model parameters.
        """
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.GET_PARAMETERS_NOT_IMPLEMENTED,
                message="Client does not implement `get_parameters`",
            ),
            parameters= Parameters(tensor_type="", tensors=[]),
        )
    

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        _ = (self, ins)
        return GetPropertiesRes(
            status=Status(
                code=Code.GET_PROPERTIES_NOT_IMPLEMENTED,
                message="Client does not implement `get_properties`",
            ),
            properties = {}
        )

    def get_context(self) -> Context:
        """Get the run context from this client."""
        return self.context

    def set_context(self, context: Context) -> None:
        """Apply a run context to this client."""
        self.context = context

    def to_client(self):
        """Return client (itself)."""
        return self
            
    def get_parameters_base(self, config):
        return [val.cpu().detach().numpy() for val in self.net.parameters()]
    
    def set_parameters(self, parameters):
        for old, new in zip(self.net.parameters(), parameters):
            old.data = torch.Tensor(new).to(DEVICE)
        
    def fit(self, ins: FitIns):
        self.set_parameters(parameters_to_ndarrays(ins.parameters))
        self.train(self.net, self.train_loader, None, self.epoch, self.lossf, self.optim, self.DEVICE, None)
        history = self.valid(self.net, self.train_loader, 0, self.lossf, self.DEVICE, False)
        return FitRes(
            status=Status(
                code=Code.OK,
                message="Client fit done",
            ),
            parameters = ndarrays_to_parameters(self.get_parameters_base({})),
            num_examples=len(self.train_loader),
            metrics = history,
        )
        # return self.get_parameters(config={}), len(self.train_loader), {}

def seeding(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

if __name__ =="__main__":
    warnings.filterwarnings("ignore")
    seeding(args)
    
    if args.data_dir != None and args.client_dir != None:
        trainset = Fets2022(args.client_dir)
        trainLoader = DataLoader(trainset, args.batch_size, True, collate_fn=lambda x: x)
    
    lossf = CustomFocalDiceLoss()
    optimizer = SGD(net.parameters(), lr=args.lr)
    
    flwr.client.start_client(
        server_address=f"{args.IPv4}:8084", client= CustomClient(net, trainLoader, args.epoch, lossf, optimizer, DEVICE, args)
    )