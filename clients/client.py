import flwr 
import torch
import numpy as np
import random

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
    
class CustomClient(flwr.client.Client):
    context: Context
    def __init__(self, net, train_loader, epoch, lossf, optimizer, DEVICE, args,trainF=lambda x: x, validF=lambda x: x):
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
            old.data = torch.Tensor(new).to(self.DEVICE)
        
    def fit(self, ins: FitIns):
        self.set_parameters(parameters_to_ndarrays(ins.parameters))
        self.train(self.net, self.train_loader, None, self.epoch, self.lossf, self.optim, self.DEVICE, None)
        return FitRes(
            status=Status(
                code=Code.OK,
                message="Client fit done",
            ),
            parameters = ndarrays_to_parameters(self.get_parameters_base({})),
            num_examples=len(self.train_loader),
            metrics = {} ,
        )
        # return self.get_parameters(config={}), len(self.train_loader), {}


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
        self.train(self.net, self.train_loader, None, self.epoch, self.lossf, self.optim, self.DEVICE, None)
        return self.get_parameters(config={}), len(self.train_loader), {}
        # return self.get_parameters(config={}), len(self.train_loader), {}

def seeding(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)