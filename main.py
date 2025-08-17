from flwr.common import Context
import server.FedAvgServer as avg
import server.FedRefServer as ref
import server.FedProxServer as prox
import server.FedOptServer as opt
import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from utils import parser
import utils.FetsTrain as fets
import utils.Cinic10Train as cinic
import utils.FEMNISTTrain as femnist
import utils.ShakespeareTrain as shakespeare
import utils.CelebaTrain as celeba
import utils.OfficeTrain as office
from utils.CustomDataset import *
from Network.Resnet import *
from Network.Unet import *
from Network.Loss import *
from Network.Mobilenet import *
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from clients import client, clientProxy, clientOpt, clientRef
import os
from torch.optim import SGD
import segmentation_models_pytorch as smp
import deeplake
import warnings

args = parser.Simulationparser()
fets.set_seeds(args)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
diceLossb = smp.losses.DiceLoss(
   mode="binary",          # For multi-class segmentation
   classes=None,               # Compute the loss for all classes
   log_loss=False,             # Do not use log version of Dice loss
   from_logits=True,           # Model outputs are raw logits
   smooth=1e-5,                # A small smoothing factor for stability
   ignore_index=None,          # Don't ignore any classes
   eps=1e-7                    # Epsilon for numerical stability
)

focalLossb = smp.losses.FocalLoss(
   mode="binary",          # Multi-class segmentation
   alpha=0.1,                 # class weighting to deal with class imbalance
   gamma=4.5,                   # Focusing parameter for hard-to-classify examples
   normalized=True
)
class CustomFocalDiceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, y):
        return diceLoss.to(DEVICE)(x, y) + focalLoss.to(DEVICE)(x, y)
    
class CustomFocalDiceLossb(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, y):
        return diceLossb.to(DEVICE)(x, y) + focalLossb.to(DEVICE)(x, y)
    
if args.mode !="fedref":
    if args.type == "fets":
        net = Custom3DUnet(1, 4, False, f_maps=4, layer_order="gcr", num_groups=4)
    if args.type == "shakespeare":
        net = ResNet(outdim=10)
    if args.type == "office":
        net = ResNet(outdim=100)
    if args.type == "femnist":
        net = MobileNet(outdim=10)
    if args.type == "cinic10":
        net = ResNet(outdim=10)
    if args.type == "celeba":
        net = ResNet(outdim=10)
    net.to(DEVICE)
    
elif args.mode =="fedref":
    if args.type == "fets":
        aggregated_net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
        aggregated_net.to(DEVICE)
        ref_net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
        ref_net.to(DEVICE)
    elif args.type == "shakespeare":
        aggregated_net = ResNet(outdim=7)
        aggregated_net.to(DEVICE)
        ref_net = ResNet(outdim=7)
        ref_net.to(DEVICE)
    elif args.type == "office":
        aggregated_net = ResNet(outdim=10)
        aggregated_net.to(DEVICE)
        ref_net = ResNet(outdim=10)
        ref_net.to(DEVICE)
    elif args.type == "cinic10":
        aggregated_net = ResNet(outdim=10)
        aggregated_net.to(DEVICE)
        ref_net = ResNet(outdim=10)
        ref_net.to(DEVICE)
    elif args.type == "femnist":
        aggregated_net = MobileNet(outdim=10)
        aggregated_net.to(DEVICE)
        ref_net = MobileNet(outdim=10)
        ref_net.to(DEVICE)
    elif args.type == "celeba":
        aggregated_net = ResNet(outdim=10)
        aggregated_net.to(DEVICE)
        ref_net = ResNet(outdim=10)
        ref_net.to(DEVICE)

if args.type == "fets":
    lossf = CustomFocalDiceLoss().to(DEVICE)
elif args.type == "office":
    lossf = focalLoss.to(DEVICE)
elif args.type == "femnist":
    lossf = nn.CrossEntropyLoss().to(DEVICE)
elif args.type == "cinic10":
    lossf = nn.CrossEntropyLoss().to(DEVICE)
elif args.type == "celeba":
    lossf = CustomFocalDiceLoss().to(DEVICE)
elif args.type == "shakespeare":
    lossf = nn.CrossEntropyLoss().to(DEVICE)

if args.type == "fets":
    if args.data_dir is None:
        pass
    else:
        valid_set = Fets2022(args.data_dir)
        validLoader = DataLoader(valid_set, args.batch_size, shuffle=False, collate_fn = lambda x: x)
elif args.type == "shakespeare":
    train_set = OCTDL(args.data_dir)
elif args.type == "celeba":
    train_set = deeplake.load('hub://activeloop/drive-train', token= args.token)
    valid_set = deeplake.load("hub://activeloop/drive-test", token= args.token)
    validLoader = DataLoader(valid_set,
        batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: {
            "rgb_images": [xi["rgb_images"].numpy() for xi in x],
            "masks": [xi["masks"].numpy() for xi in x]
        }
    )
elif args.type == "femnist":
    train_set = datasets.MNIST("./Data", True, Compose([ToTensor(), Resize((64,64))]), None, True)
    valid_set = datasets.MNIST("./Data", False, Compose([ToTensor(), Resize((64,64))]), None, True)
    validLoader = DataLoader(valid_set, args.batch_size, shuffle=False, collate_fn = lambda x: x)
elif args.type == "cinic10":
    train_set = datasets.CIFAR10("./Data", True, Compose([ToTensor(), Resize((64,64)), Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]), None, True)
    valid_set = datasets.CIFAR10("./Data", False, Compose([ToTensor(), Resize((64,64)), Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]), None, True)
    validLoader = DataLoader(valid_set, args.batch_size, shuffle=False, collate_fn = lambda x: x)
elif args.type == "office":
    train_set = datasets.CIFAR10("./Data", True, Compose([ToTensor(), Resize((64,64)), Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]), None, True)
    valid_set = datasets.CIFAR10("./Data", False, Compose([ToTensor(), Resize((64,64)), Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]), None, True)
    validLoader = DataLoader(valid_set, args.batch_size, shuffle=False, collate_fn = lambda x: x)

if args.type == "fets":
    client_dirs = [os.path.join(args.client_dir, f"client{num}") for num in range(1, 11)]

def set_parameters(net, new_parameters):
    for old, new in zip(net.parameters(), new_parameters):
        shape = old.data.size()
        old.data = torch.Tensor(new).view(shape).to(DEVICE)

def client_fn(context: Context):
    id = int(context.node_id)%10
    if args.type == "fets":
        trainset = Fets2022(client_dirs[id])
        train_loader = DataLoader(trainset, args.batch_size, shuffle=True, collate_fn=lambda x:x)
    else:
        trainset = train_set
    
    if args.type == "celeba":
        trainF = celeba.train
        validF = celeba.valid
    elif args.type == "fets" :
        trainF = fets.train
        validF = fets.valid
    elif args.type == "femnist":
        trainF = femnist.train
        validF = femnist.valid
    elif args.type == "cinic10":
        trainF = cinic.train
        validF = cinic.valid
    elif args.type == "office":
        trainF = office.train
        validF = office.valid
    elif args.type == "shakespeare":
        trainF = shakespeare.train
        validF = shakespeare.valid

    if args.mode == "fedref":
        return clientRef.CustomNumpyClient(aggregated_net, train_loader, args.epoch, lossf, SGD(aggregated_net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode == "fedavg":
        return client.CustomNumpyClient(net, train_loader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode == "fedprox":
        return clientProxy.CustomNumpyClient(net, train_loader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode == "fedopt":
        return clientOpt.CustomNumpyClient(net, train_loader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Please choose from ['fedavg', 'fedref', 'fedprox', 'fedopt', 'fedyogi', 'fedadam', 'fedadagrad'].")
    
    

if __name__ =="__main__":
    warnings.filterwarnings("ignore")
    fets.make_model_folder(f"./Models/{args.version}")
    
    if args.mode =="fedavg":
        strategy = avg.FedAvg(net, lossf, validLoader, args, inplace=True, evaluate_fn=lambda p, c: c,  min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num)
    elif args.mode =="fedref":
        strategy = ref.FedRef(ref_net, aggregated_net, lossf, validLoader, args, args.prime,evaluate_fn=lambda p, c: c, inplace=False, min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num)
    elif args.mode =="fedprox":
        strategy = prox.FedProx(net, lossf, validLoader, args, proximal_mu=0.5, evaluate_fn=lambda p, c: c,inplace=False, min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num)
    elif args.mode =="fedopt":
        strategy = opt.FedOpt(net, lossf, validLoader, args, initial_parameters=[layer.cpu().detach().numpy() for layer in net.parameters()], min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num, evaluate_fn=lambda p, c: c)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Please choose from ['fedavg', 'fedref', 'fedprox', 'fedopt', 'fedyogi', 'fedadam', 'fedadagrad'].")
    
    def server_fn(context):
        return fl.server.ServerAppComponents(strategy= strategy, config=fl.server.ServerConfig(args.round))
    
    if args.gpu:
        fl.simulation.run_simulation(
     client_app= fl.client.ClientApp(client_fn=client_fn),
     server_app= fl.server.ServerApp(server_fn=server_fn),
     num_supernodes= args.client_num,
     backend_config={"client_resources": {"num_cpus": 1.0 , "num_gpus": 1}},
     verbose_logging=False
    )
    else:
        fl.simulation.run_simulation(
     client_app= fl.client.ClientApp(client_fn=client_fn),
     server_app= fl.server.ServerApp(server_fn=server_fn),
     num_supernodes= args.client_num,
     backend_config={"client_resources": {"num_cpus": 1.0, "num_gpus": 0}},
     verbose_logging=False
    )
    