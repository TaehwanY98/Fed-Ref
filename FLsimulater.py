from flwr.common import Context
import server.FedAvgServer as avg
import server.FedLWRServer as lwr
import server.FedPIDServer as pid
import server.FedRefServer as ref
import flwr as fl
import torch
from torch.utils.data import DataLoader
from utils import parser
import utils.train as seg
import utils.octTrain as oct
from utils.CustomDataset import Fets2022, BRATS, OCTDL
from Network.Resnet import *
from Network.Unet import *
from Network.Loss import *
from clients import client, clientPID
import os
from torch.optim import SGD
import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parser.Simulationparser()
seg.set_seeds(args)

diceLoss = smp.losses.DiceLoss(
   mode="multiclass",          # For multi-class segmentation
   classes=None,               # Compute the loss for all classes
   log_loss=False,             # Do not use log version of Dice loss
   from_logits=True,           # Model outputs are raw logits
   smooth=1e-5,                # A small smoothing factor for stability
   ignore_index=None,          # Don't ignore any classes
   eps=1e-7                    # Epsilon for numerical stability
)

focalLoss = smp.losses.FocalLoss(
   mode="multiclass",          # Multi-class segmentation
   alpha=0.1,                 # class weighting to deal with class imbalance
   gamma=4.5,                   # Focusing parameter for hard-to-classify examples
   normalized=True
)
class CustomFocalDiceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, y):
        return diceLoss.to(DEVICE)(x, y) + focalLoss.to(DEVICE)(x, y)

if args.mode !="fedref":
    if args.type == "fets":
        net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
    if args.type == "brats":
        net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
    if args.type == "octdl":
        net = ResNet()
    net.to(DEVICE)
elif args.mode =="fedref":
    if args.type in ["fets", "brats"]:
        aggregated_net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
        aggregated_net.to(DEVICE)
        ref_net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
        ref_net.to(DEVICE)
    elif args.type in "octdl":
        aggregated_net = ResNet()
        aggregated_net.to(DEVICE)
        ref_net = ResNet()
        ref_net.to(DEVICE)
if args.type == "fets":
    dataset = Fets2022(args.data_dir)
if args.type == "brats":
    dataset = BRATS(args.data_dir)
if args.type == "octdl":
    dataset = OCTDL(args.data_dir)
validLoader = DataLoader(dataset, args.batch_size, False, collate_fn=lambda x: x)
if args.type == "fets":
    client_dirs = [os.path.join(args.client_dir, f"client{num}") for num in range(1, 11)]
if args.type == "brats":
    client_dirs = [os.path.join(args.client_dir, f"{num}") for num in range(1, 11)]
if args.type == "octdl":
    client_dirs = [os.path.join(args.client_dir, f"{num}") for num in range(1, 11)]
def set_parameters(net, new_parameters):
    for old, new in zip(net.parameters(), new_parameters):
        shape = old.data.size()
        old.data = torch.Tensor(new).view(shape).to(DEVICE)

def save_result(args):
    if args.type == 'fets':
        pass
    elif args.type == "brats":
        pass
    elif args.type == "octdl":
        pass


def client_fn(context: Context):
    id = int(context.node_id)%10
    if args.type == "fets":
        trainset = Fets2022(client_dirs[id])
    if args.type == "brats":
        trainset = BRATS(client_dirs[id])
    if args.type == "octdl":
        trainset = OCTDL(client_dirs[id])
    train_loader = DataLoader(trainset, args.batch_size, shuffle=True, collate_fn=lambda x: x)
    if args.type == "octdl":
        lossf = nn.BCEWithLogitsLoss(trainset.label_weight)
        trainF = oct.train
        validF = oct.valid
    else :
        lossf = CustomFocalDiceLoss()
        trainF = seg.train
        validF = seg.valid 
    if args.mode == "fedpid":
        return clientPID.CustomNumpyClient(net, train_loader, validLoader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode == "fedref":
        return client.CustomNumpyClient(aggregated_net, train_loader, args.epoch, lossf, SGD(aggregated_net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    else :
        return client.CustomNumpyClient(net, train_loader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    


if __name__ =="__main__":
    seg.warnings.filterwarnings("ignore")
    seg.make_model_folder(f"./Models/{args.version}")
    if args.mode =="fedavg":
        strategy = avg.FedAvg(net, dataset, validLoader, args, inplace=True, evaluate_fn=lambda p, c: c,  min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num)
        
    elif args.mode =="fedpid":
        strategy = pid.FedPID(net, dataset, validLoader, args, evaluate_fn=lambda p, c: c,inplace=False, min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num)
        
    elif args.mode =="fedlwr":
        strategy = lwr.FedLWR(net, dataset, validLoader, args, evaluate_fn=lambda p, c: c,inplace=False, min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num)
        
    elif args.mode =="fedref":
        strategy = ref.FedRef(ref_net, aggregated_net, dataset, validLoader, args, evaluate_fn=lambda p, c: c, inplace=False, min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num)
        

    def server_fn(context):
        return fl.server.ServerAppComponents(strategy= strategy, config=fl.server.ServerConfig(args.round))
    
    if args.gpu:
        fl.simulation.run_simulation(
     client_app= fl.client.ClientApp(client_fn=client_fn),
     server_app= fl.server.ServerApp(server_fn=server_fn),
     num_supernodes= args.client_num,
     backend_config={"client_resources": {"num_cpus": args.client_num , "num_gpus": 1}},
     verbose_logging=False
    )
    else:
        fl.simulation.run_simulation(
     client_app= fl.client.ClientApp(client_fn=client_fn),
     server_app= fl.server.ServerApp(server_fn=server_fn),
     num_supernodes= args.client_num,
     backend_config={"client_resources": {"num_cpus": args.client_num, "num_gpus": 0}},
     verbose_logging=False
    )
    