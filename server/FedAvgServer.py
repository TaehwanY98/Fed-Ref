from typing import Dict, List, Optional, Tuple
import flwr
import torch
import warnings
from torch.utils.data import DataLoader
from torch import save
from utils.train import valid , make_model_folder, set_seeds, CustomFocalDiceLossb, CustomFocalDiceLoss,validDrive
from utils.octTrain import valid as octValid
from utils.MNISTTrain import valid as MNISTValid
from utils.CIFAR10Train import valid as CIFAR10Valid
from utils.parser import Federatedparser
from utils.CustomDataset import Fets2022, BRATS, OCTDL
from Network.Resnet import *
from Network.Unet import *
from Network.Loss import *
import pandas as pd
import os
from flwr.common import (
    parameters_to_ndarrays,
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedAvg(flwr.server.strategy.FedAvg):
    def __init__(self, net, lossf, validLoader, args, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.net = net
        self.args = args
        self.lossf = lossf
        self.validLoader = validLoader
        self.evaluate_fn = self.evaluate_fn
    def aggregate_fit(self, server_round, results, failures):
        return super().aggregate_fit(server_round, results, failures)
    def evaluate(self, server_round: int, parameters)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
        parameters = parameters_to_ndarrays(parameters)
        validF= valid if not self.args.type=="octdl" else octValid
        if self.args.type=="mnist":
            validF = MNISTValid
        if self.args.type == "cifar10":
            validF = CIFAR10Valid
        if self.args.type in ["drive"]:
            validF = validDrive
        set_parameters(self.net, parameters)
        history=validF(self.net, self.validLoader, 0, self.lossf.to(DEVICE), DEVICE, True)
        make_dir(self.args.result_path)
        make_dir(os.path.join(self.args.result_path, self.args.mode))
        if server_round != 0:
            old_historyframe = pd.read_csv(os.path.join(self.args.result_path, self.args.mode, f'FedAvg_{self.args.type}.csv'))
            historyframe = pd.DataFrame({k:[v] for k, v in history.items()})
            newframe=pd.concat([old_historyframe, historyframe])
            newframe.to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedAvg_{self.args.type}.csv'), index=False)
        else:
            pd.DataFrame({k:[v] for k, v in history.items()}).to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedAvg_{self.args.type}.csv'), index=False)
        save(self.net.state_dict(), f"./Models/{self.args.version}/net.pt")
        return history['loss'], {key:value for key, value in history.items() if key != "loss" }
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def set_parameters(net, parameters):
    for old, new in zip(net.parameters(), parameters):
        old.data = torch.Tensor(new).to(DEVICE)

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    args = Federatedparser()
    make_model_folder(f"./Models/{args.version}")
    set_seeds(args)
    
    if args.type == "fets":
        dataset = Fets2022(args.client_dir)
        net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
    if args.type == "brats":
        dataset = BRATS(args.client_dir)
        net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
    if args.type == "octdl":
        dataset = OCTDL(args.client_dir)
        net = ResNet()
    net.to(DEVICE)
    validLoader = DataLoader(dataset, args.batch_size, False, collate_fn=lambda x: x)
 
    def fl_save(server_round:int, parameters: flwr.common.NDArrays, config:Dict[str, flwr.common.Scalar], validF=valid)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
        set_parameters(net, parameters)
        save(net.state_dict(), f"./Models/{args.version}/net.pt")
        print("model is saved")
        return 0, {}

    def fl_evaluate(server_round:int, parameters: flwr.common.NDArrays, config:Dict[str, flwr.common.Scalar], validF= valid if not args.type=="octdl" else octValid, lossf = CustomFocalDiceLoss() if not args.type=="octdl" else nn.BCEWithLogitsLoss(dataset.label_weight))-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
        set_parameters(net, parameters)
        history=validF(net, validLoader, 0, lossf.to(DEVICE), DEVICE, True)
        save(net.state_dict(), f"./Models/{args.version}/net.pt")
        return history['loss'], {key:value for key, value in history.items() if key != "loss" }
    
    history = flwr.server.start_server(
        server_address='[::]:8084',strategy=FedAvg(net, dataset, validLoader, evaluate_fn=fl_evaluate, inplace=True, min_fit_clients=7, min_available_clients=7, min_evaluate_clients=7), 
                           config=flwr.server.ServerConfig(num_rounds=args.round)
    )
    if args.type == 'fets':
        pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedAvg_fets.csv", index=False)
        pd.DataFrame(history.losses_centralized).to_csv("./Result/FedAvg_loss_fets.csv", index=False)
    elif args.type == "brats":
        pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedAvg_BRATS.csv", index=False)
        pd.DataFrame(history.losses_centralized).to_csv("./Result/FedAvg_loss_BRATS.csv", index=False)
    elif args.type == "octdl":
        pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedRef_OCTDL.csv", index=False)
        pd.DataFrame(history.losses_centralized).to_csv("./Result/FedRef_loss_OCTDL.csv", index=False)