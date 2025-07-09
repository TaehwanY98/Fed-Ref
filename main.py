from flwr.common import Context
import server.FedAvgServer as avg
import server.FedRefServer as ref
import server.FedProxServer as prox
import server.FedOptServer as opt
import server.FedAdagradServer as adagrad
import server.FedAdamServer as adam
import server.FedYogiServer as yogi
import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from utils import parser
import utils.TumorTrain as seg
import utils.octTrain as oct
import utils.MNISTTrain as mnist
import utils.CIFAR10Train as cifar10
from utils.CustomDataset import *
from Network.Resnet import *
from Network.Unet import *
from Network.Loss import *
from Network.Mobilenet import *
from clients import client, clientProxy, clientOpt, clientAdam, clientAdagrad, clientRef, clientYogi
import os
from torch.optim import SGD
import segmentation_models_pytorch as smp
import deeplake
import random
import warnings
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
        net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
    if args.type == "brats":
        net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
    if args.type == "octdl":
        net = ResNet(outdim=7)
    if args.type == "mnist":
        net = MobileNet(outdim=10)
    if args.type == "cifar10":
        net = ResNet(outdim=10)
    if args.type == "drive":
        net = Custom2DUnet(3, 1, True, f_maps=4, layer_order="cr", num_groups=4)
    net.to(DEVICE)
    
elif args.mode =="fedref":
    if args.type in ["fets", "brats"]:
        aggregated_net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
        aggregated_net.to(DEVICE)
        ref_net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
        ref_net.to(DEVICE)
    elif args.type == "octdl":
        aggregated_net = ResNet(outdim=7)
        aggregated_net.to(DEVICE)
        ref_net = ResNet(outdim=7)
        ref_net.to(DEVICE)
    elif args.type == "mnist":
        aggregated_net = MobileNet(outdim=10)
        aggregated_net.to(DEVICE)
        ref_net = MobileNet(outdim=10)
        ref_net.to(DEVICE)
    elif args.type == "cifar10":
        aggregated_net = ResNet(outdim=10)
        aggregated_net.to(DEVICE)
        ref_net = ResNet(outdim=10)
        ref_net.to(DEVICE)
        
    elif args.type in ["drive"]:
        aggregated_net = Custom2DUnet(3, 1, True, f_maps=4, layer_order="cr", num_groups=4)
        aggregated_net.to(DEVICE)
        ref_net = Custom2DUnet(3, 1, True, f_maps=4, layer_order="cr", num_groups=4)
        ref_net.to(DEVICE)

if args.type == "octdl":
    lossf = nn.BCEWithLogitsLoss().to(DEVICE)
elif args.type in ["fets", "brats"] :
    lossf = CustomFocalDiceLoss().to(DEVICE)
elif args.type in ["drive"] :
    lossf = CustomFocalDiceLossb().to(DEVICE)
elif args.type == "mnist":
    lossf = nn.CrossEntropyLoss().to(DEVICE)
elif args.type == "cifar10":
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="mean").to(DEVICE)


if args.type == "fets":
    train_set = Fets2022(args.data_dir)
elif args.type == "brats":
    train_set = BRATS(args.data_dir)
elif args.type == "octdl":
    train_set = OCTDL(args.data_dir)
elif args.type == "drive":
    train_set = deeplake.load("hub://activeloop/drive-train")
elif args.type == "mnist":
    train_set = datasets.MNIST("./Data", True, Compose([ToTensor(), Resize((64,64))]), None, True)
    valid_set = datasets.MNIST("./Data", False, Compose([ToTensor(), Resize((64,64))]), None, True)
    validLoader = DataLoader(valid_set, args.batch_size, shuffle=False, collate_fn = lambda x: x)
elif args.type == "cifar10":
    train_set = datasets.CIFAR10("./Data", True, Compose([ToTensor(), Resize((64,64))]), None, True)
    valid_set = datasets.CIFAR10("./Data", False, Compose([ToTensor(), Resize((64,64))]), None, True)
    validLoader = DataLoader(valid_set, args.batch_size, shuffle=False, collate_fn = lambda x: x)
if args.type == "fets":
    client_dirs = [os.path.join(args.client_dir, f"client{num}") for num in range(1, 11)]
if args.type == "brats":
    client_dirs = [os.path.join(args.client_dir, f"{num}") for num in range(1, 11)]
if args.type == "octdl":
    client_dirs = [os.path.join(args.client_dir, f"{num}") for num in range(1, 11)]
if args.type == "mnist":
    pass
if args.type == "cifar10":
    pass
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
    if args.type == "drive":
        trainset = train_set
    if args.type in ["mnist", 'cifar10']:
        trainset = train_set
    if args.type in ["fets","brats", "octdl"]:
        train_loader = DataLoader(trainset, args.batch_size, shuffle=True, collate_fn=lambda x: x)
    if args.type in ["drive", "mnist", "cifar10"]:
        if args.test:
            i = 0.01
        else:
            i = random.randint(5, 9)/10
            
        trainS, _ = random_split(trainset, [i, 1-i], torch.Generator("cpu").manual_seed(args.seed))
        train_loader = DataLoader(trainS, args.batch_size, shuffle=True, collate_fn=lambda x: x)
    if args.type == "octdl":
        trainF = oct.train
        validF = oct.valid
    elif args.type in ["fets", "brats"] :
        trainF = seg.train
        validF = seg.valid
    elif args.type in ["drive"] :
        trainF = seg.trainDrive
        validF = seg.validDrive
    elif args.type == "mnist":
        trainF = mnist.train
        validF = mnist.valid
    elif args.type == "cifar10":
        trainF = cifar10.train
        validF = cifar10.valid
    if args.mode in ["fedref"]:
        return clientRef.CustomNumpyClient(aggregated_net, train_loader, args.epoch, lossf, SGD(aggregated_net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode in ["fedavg"]:
        return client.CustomNumpyClient(net, train_loader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode in ["fedprox"]:
        return clientProxy.CustomNumpyClient(net, train_loader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode in ["fedopt"]:
        return clientOpt.CustomNumpyClient(net, train_loader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode in ["fedyogi"]:
        return clientYogi.CustomNumpyClient(net, train_loader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode in ["fedadam"]:
        return clientAdam.CustomNumpyClient(net, train_loader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode in ["fedadagrad"]:
        return clientAdagrad.CustomNumpyClient(net, train_loader, args.epoch, lossf, SGD(net.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Please choose from ['fedavg', 'fedref', 'fedprox', 'fedopt', 'fedyogi', 'fedadam', 'fedadagrad'].")
    
    

if __name__ =="__main__":
    warnings.filterwarnings("ignore")
    seg.make_model_folder(f"./Models/{args.version}")
    if args.mode =="fedavg":
        strategy = avg.FedAvg(net, lossf, validLoader, args, inplace=True, evaluate_fn=lambda p, c: c,  min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num)
        
    elif args.mode =="fedref":
        strategy = ref.FedRef(ref_net, aggregated_net, lossf, validLoader, args, args.prime, evaluate_fn=lambda p, c: c, inplace=False, min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num)
        
    elif args.mode =="fedprox":
        strategy = prox.FedProx(net, lossf, validLoader, args, proximal_mu=0.5, evaluate_fn=lambda p, c: c,inplace=False, min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num)
    elif args.mode =="fedopt":
        strategy = opt.FedOpt(net, lossf, validLoader, args, initial_parameters=[layer.cpu().detach().numpy() for layer in net.parameters()], min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num, evaluate_fn=lambda p, c: c)
    elif args.mode =="fedyogi":
        strategy = yogi.FedYogi(net, lossf, validLoader, args, initial_parameters=[layer.cpu().detach().numpy() for layer in net.parameters()], min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num, evaluate_fn=lambda p, c: c)
    elif args.mode =="fedadam":
        strategy = adam.FedAdam(net, lossf, validLoader, args, initial_parameters=[layer.cpu().detach().numpy() for layer in net.parameters()], min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num, evaluate_fn=lambda p, c: c)
    elif args.mode =="fedadagrad":
        strategy = adagrad.FedAdagrad(net, lossf, validLoader, args, initial_parameters=[layer.cpu().detach().numpy() for layer in net.parameters()], min_fit_clients=args.client_num, min_available_clients=args.client_num, min_evaluate_clients=args.client_num, evaluate_fn=lambda p, c: c)
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
    