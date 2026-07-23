from flwr.common import Context
import server.FedAvgServer as avg
import server.FedRefServer as ref
import server.FedProxServer as prox
import server.FedOptServer as opt
import server.FedEveServer as eve
import server.AdaBestSever as adabest
import server.AFedPDServer as afedpd

import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, GaussianBlur, RandomApply, ToPILImage
from torchvision.transforms.v2 import GaussianNoise

# from torchvision.transforms import Compose, ToTensor, 
from utils import parser
import utils.FetsTrain as fets
import utils.Cinic10Train as cinic
import utils.FEMNISTTrain as femnist
import utils.ShakespeareTrain as shakespeare
import utils.CelebaTrain as celeba
import utils.OfficeTrain as office
from utils.CustomDataset import *
from Network.CNN import *
from Network.Resnet import *
from Network.Unet import *
from Network.Loss import *
from Network.Mobilenet import *
from clients import client, clientProxy, clientOpt, clientRef, clientAdaBest, clientFedEve, clientAFedPD
import os
from torch.optim import SGD
import segmentation_models_pytorch as smp
import warnings
import datasets
import copy

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)

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

class CustomFocalDiceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, y):
        return diceLoss.to(DEVICE)(x, y) + focalLoss.to(DEVICE)(x, y)
    
if args.mode !="fedref":
    if args.type == "fets":
        net = Custom3DUnet(1, 4, False, f_maps=4, layer_order="gcr", num_groups=4)
    if args.type == "shakespeare":
        net = ResNet(outdim=10)
    if args.type == "office":
        net = ResNet(outdim=100)
    if args.type == "femnist":
        net = FEMNIST_CNN(num_classes=62)
    if args.type == "cinic10":
        net = CINIC10_LightCNN(num_classes=10)
    if args.type == "celeba":
        net = ResNet(outdim=2)
    net.to(DEVICE)
    
elif args.mode =="fedref":
    if args.type == "fets":
        aggregated_net = Custom3DUnet(1, 4, False, f_maps=4, layer_order="gcr", num_groups=4)
        aggregated_net.to(DEVICE)
        ref_net = Custom3DUnet(1, 4, False, f_maps=4, layer_order="gcr", num_groups=4)
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
        aggregated_net = CINIC10_LightCNN(num_classes=10)
        aggregated_net.to(DEVICE)
        ref_net = CINIC10_LightCNN(num_classes=10)
        ref_net.to(DEVICE)
    elif args.type == "femnist":
        aggregated_net = FEMNIST_CNN(num_classes=62)
        aggregated_net.to(DEVICE)
        ref_net = FEMNIST_CNN(num_classes=62)
        ref_net.to(DEVICE)
    elif args.type == "celeba":
        aggregated_net = ResNet(outdim=2)
        aggregated_net.to(DEVICE)
        ref_net = ResNet(outdim=2)
        ref_net.to(DEVICE)

if args.type == "fets":
    lossf = CustomFocalDiceLoss().to(DEVICE)
elif args.type == "office":
    lossf = nn.CrossEntropyLoss().to(DEVICE)
elif args.type == "femnist":
    lossf = AsymmetricLoss().to(DEVICE)
elif args.type == "cinic10":
    lossf = AsymmetricLoss().to(DEVICE)
elif args.type == "celeba":
    lossf = AsymmetricLoss().to(DEVICE)
elif args.type == "shakespeare":
    lossf = nn.CrossEntropyLoss().to(DEVICE)

if args.type == "fets":
    if args.data_dir is None:
        pass
    else:
        valid_set = Fets2022(args.data_dir, args.degrade)
        validLoader = DataLoader(valid_set, args.batch_size, shuffle=False, collate_fn = lambda x: x)

elif args.type == "femnist":
    def FlippingAttack(example):
        for i, e in enumerate(example["image"]):
            example['character'][i] = torch.randint(low=0, high=61, size=(1,)).cpu().item()
        return example
    
    def CustomTransform(example, rank):
        if len(example) <= 1:
            out=ToTensor()(example["image"])
            out=RandomApply([GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), GaussianNoise()], p=args.degrade)(out)
            out = ToPILImage()(out.cpu().detach())
            example["image"] = out
        else:
            for i, e in enumerate(example["image"]):
                out=ToTensor()(e)
                out=RandomApply([GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), GaussianNoise()], p=args.degrade)(out)
                out = ToPILImage()(out.cpu().detach())
                example["image"][i] = out
        return RandomApply([FlippingAttack], p=args.degrade)(example)
        
    Femnist = datasets.load_dataset("flwrlabs/femnist")
    data_set = Femnist["train"]
    data_set = data_set.train_test_split(test_size=0.1, seed=args.seed)
    validLoader = data_set["test"].shuffle(args.seed).to_iterable_dataset().batch(args.batch_size)
    # data_set = data_set["train"].map(CustomTransform, batched=True, batch_size=args.batch_size, num_proc=1, with_rank=True)
    data_set = data_set["train"]
    info = {"num_samples": data_set.to_pandas()["hsf_id"].value_counts().sort_index()}

elif args.type == "cinic10":
    
    def FlippingAttack(example):
        for i, e in enumerate(example["image"]):
            example['label'][i] = torch.randint(low=0, high=9, size=(1,)).cpu().item()
        return example
    
    def CustomTransform(example, rank):
        if len(example) <= 1:
            out=ToTensor()(example["image"])
            out=RandomApply([GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), GaussianNoise()], p=args.degrade)(out)
            out = ToPILImage()(out.cpu().detach())
            example["image"] = out
        else:
            for i, e in enumerate(example["image"]):
                out=ToTensor()(e)
                out=RandomApply([GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), GaussianNoise()], p=args.degrade)(out)
                out = ToPILImage()(out.cpu().detach())
                example["image"][i] = out
        return RandomApply([FlippingAttack], p=args.degrade)(example)
    
    CINIC10 = datasets.load_dataset("flwrlabs/cinic10")
    # data_set = CINIC10["train"].map(CustomTransform, batched=True, batch_size=args.batch_size, num_proc=1, with_rank=True)
    data_set = CINIC10["train"]
    validLoader = CINIC10["test"].shuffle(args.seed).to_iterable_dataset().batch(args.batch_size)
    info = {"num_samples": [9000]*10}


if args.type == "fets":
    client_dirs = [os.path.join(args.client_dir, f"client{num}") for num in range(1, 17)]
elif args.type == "shakespeare":
    pass
elif args.type == "celeba":
    dataset_partions = [data_set.to_iterable_dataset().filter(lambda x:x["celeb_id"]%16==id) for id in range(0, 16)]
elif args.type == "femnist":
    dataset_partions = [data_set.skip(sum(info["num_samples"].values[:idx])).take(id).to_iterable_dataset() if idx!=0 else data_set.take(id).to_iterable_dataset() for idx, id in enumerate(info["num_samples"].values)]
elif args.type == "cinic10":
    dataset_partions = [data_set.shuffle(args.seed).skip(sum(info["num_samples"][:idx])).take(id).to_iterable_dataset() if idx!=0 else data_set.shuffle(args.seed).take(id).to_iterable_dataset() for idx, id in enumerate(info["num_samples"])]

def client_fn(cid: str):
    if torch.rand(size=1).item() < args.degrade:
        learning_rates = [1e-2, 1e-3, 1e-4, 9e-6, 4e-6, 1e-6]
        args.lr = learning_rates[torch.randint(0, len(learning_rates)-1, size=1).item()]
    if args.type == "fets":
        id = int(cid) % 15
        trainset = Fets2022(client_dirs[id])
        train_loader = DataLoader(trainset, args.batch_size, shuffle=True, collate_fn=lambda x:x)
        length = len(train_loader)
        trainF = fets.train
        validF = fets.valid
    elif args.type == "shakespeare":
        trainF = shakespeare.train
        validF = shakespeare.valid
    elif args.type == "celeba":
        id = int(cid) % 16
        length = int(info["custom_part"][id] // args.batch_size)
        train_loader = dataset_partions[id].shuffle(buffer_size=1000, seed=args.seed).batch(args.batch_size)
        trainF = celeba.train
        validF = celeba.valid
    elif args.type == "femnist":
        id = int(cid) % 7
        length = int(info["num_samples"].iloc[id] // args.batch_size)
        train_loader = dataset_partions[id].shuffle(buffer_size=1000, seed=args.seed).batch(args.batch_size)
        trainF = femnist.train
        validF = femnist.valid
    elif args.type == "cinic10":
        id = int(cid) % 10
        length = int(info["num_samples"][id] // args.batch_size)
        train_loader = dataset_partions[id].shuffle(buffer_size=1000).batch(args.batch_size)
        trainF = cinic.train
        validF = cinic.valid
    elif args.type == "office":
        trainF = office.train
        validF = office.valid
    if args.mode == "fedref":
        net_instance = copy.deepcopy(aggregated_net)
        return clientRef.CustomNumpyClient(cid, net_instance, train_loader, length, args.epoch, lossf, SGD(net_instance.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode == "fedavg":
        net_instance = copy.deepcopy(net)
        return client.CustomNumpyClient(net_instance, train_loader, length, args.epoch, lossf, SGD(net_instance.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode == "fedprox":
        net_instance = copy.deepcopy(net)
        return clientProxy.CustomNumpyClient(net_instance, train_loader, length,args.epoch, lossf, SGD(net_instance.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode == "fedopt":
        net_instance = copy.deepcopy(net)
        return clientOpt.CustomNumpyClient(net_instance, train_loader, length,args.epoch, lossf, SGD(net_instance.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode == "adabest":
        net_instance = copy.deepcopy(net)
        return clientAdaBest.CustomNumpyClient(cid, net_instance, train_loader, length,args.epoch, lossf, SGD(net_instance.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode == "fedeve":
        net_instance = copy.deepcopy(net)
        return clientFedEve.CustomNumpyClient(cid, net_instance, train_loader, length,args.epoch, lossf, SGD(net_instance.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    elif args.mode == "afedpd":
        net_instance = copy.deepcopy(net)
        return clientAFedPD.CustomNumpyClient(cid, net_instance, train_loader, length,args.epoch, lossf, SGD(net_instance.parameters(), args.lr), DEVICE, args, trainF, validF).to_client()
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Please choose from ['fedavg', 'fedref', 'fedprox', 'fedopt', 'adabest', 'fedeve'].")
    

if __name__ =="__main__":
    warnings.filterwarnings("ignore")
    fets.make_model_folder(f"./Models/{args.version}")
    
    if args.mode =="fedavg":
        strategy = avg.FedAvg(net, lossf, validLoader, args, inplace=True, evaluate_fn=lambda p, c: c,  min_fit_clients=args.numberOfNodes, min_available_clients=args.numberOfNodes, min_evaluate_clients=args.numberOfNodes)
    elif args.mode =="fedref":
        strategy = ref.FedRef(ref_net, aggregated_net, lossf, validLoader, args, args.prime,evaluate_fn=lambda p, c: c, inplace=False, min_fit_clients=args.numberOfNodes, min_available_clients=args.numberOfNodes, min_evaluate_clients=args.numberOfNodes)
    elif args.mode =="fedprox":
        strategy = prox.FedProx(net, lossf, validLoader, args, proximal_mu=0.5, evaluate_fn=lambda p, c: c,inplace=False, min_fit_clients=args.numberOfNodes, min_available_clients=args.numberOfNodes, min_evaluate_clients=args.numberOfNodes)
    elif args.mode =="fedopt":
        strategy = opt.FedOpt(net, lossf, validLoader, args, initial_parameters=[layer.cpu().detach().numpy() for layer in net.parameters()], min_fit_clients=args.numberOfNodes, min_available_clients=args.numberOfNodes, min_evaluate_clients=args.numberOfNodes, evaluate_fn=lambda p, c: c, eta=1e-2, beta_1=0.9, beta_2=0.99, tau=1e-4)
    elif args.mode == "adabest":
        strategy = adabest.AdaBest(net, lossf, validLoader, args,evaluate_fn=lambda p, c: c, min_fit_clients=args.numberOfNodes, min_available_clients=args.numberOfNodes, min_evaluate_clients=args.numberOfNodes)
    elif args.mode == "fedeve":
        strategy = eve.FedEve(net, lossf, validLoader, args, evaluate_fn=lambda p, c: c, min_fit_clients=args.numberOfNodes, min_available_clients=args.numberOfNodes, min_evaluate_clients=args.numberOfNodes)
    elif args.mode == "afedpd":
        strategy = afedpd.A_FedPD(net, lossf, validLoader, args, evaluate_fn=lambda p, c: c,  min_fit_clients=args.numberOfNodes, min_available_clients=args.numberOfNodes, min_evaluate_clients=args.numberOfNodes)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Please choose from ['fedavg', 'fedref', 'fedprox', 'fedopt', 'adabest', 'fedeve'].")
    
    def server_fn(context):
        return fl.server.ServerAppComponents(strategy= strategy, config=fl.server.ServerConfig(args.round))
    
    if args.gpu:
        fl.simulation.run_simulation(
     client_app= fl.client.ClientApp(client_fn=client_fn),
     server_app= fl.server.ServerApp(server_fn=server_fn),
     num_supernodes= args.numberOfNodes,
     backend_config={"client_resources": {"num_cpus": 1.0 , "num_gpus": 1}},
     verbose_logging=False
    )
    else:
        fl.simulation.run_simulation(
     client_app= fl.client.ClientApp(client_fn=client_fn),
     server_app= fl.server.ServerApp(server_fn=server_fn),
     num_supernodes= args.numberOfNodes,
     backend_config={"client_resources": {"num_cpus": 1.0, "num_gpus": 0}},
     verbose_logging=False
    )
    