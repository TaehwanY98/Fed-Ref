from typing import Dict, List, Optional, Tuple
import flwr
import torch
from torch.utils.data import DataLoader
from torch import save
from train import valid , make_model_folder, set_seeds, CustomFocalDiceLoss
import warnings
from utils.parser import Federatedparser
from utils.CustomDataset import Fets2022
from Network.Unet import *
from Network.Loss import *
import sys
import getpass
user = getpass.getuser()
sys.path.append(f"/home/{user}/MICCAI/Network")
from client import seeding

import pandas as pd


args = Federatedparser()
seeding(args)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
net.to(DEVICE)

dataset = Fets2022(args.client_dir)
validLoader = DataLoader(dataset, args.batch_size, False, collate_fn=lambda x: x)

class FedAvg(flwr.server.strategy.FedAvg):
    def __init__(self, *, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        
    def aggregate_fit(self, server_round, results, failures):
        return super().aggregate_fit(server_round, results, failures)

def set_parameters(net, parameters):
    for old, new in zip(net.parameters(), parameters):
        old.data = torch.Tensor(new).to(DEVICE)

def fl_save(server_round:int, parameters: flwr.common.NDArrays, config:Dict[str, flwr.common.Scalar], validF=valid)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
    set_parameters(net, parameters)
    save(net.state_dict(), f"./Models/{args.version}/net.pt")
    print("model is saved")
    return 0, {}

def fl_evaluate(server_round:int, parameters: flwr.common.NDArrays, config:Dict[str, flwr.common.Scalar], validF=valid)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
    set_parameters(net, parameters)
    history=validF(net, validLoader, 0, CustomFocalDiceLoss(), DEVICE)
    save(net.state_dict(), f"./Models/{args.version}/net.pt")
    return history['loss'], {key:value for key, value in history.items() if key != "loss" }

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    make_model_folder(f"./Models/{args.version}")
    seeding(args)
    
    history = flwr.server.start_server(
        server_address='[::]:8084',strategy=FedAvg(evaluate_fn=fl_evaluate, inplace=True, min_fit_clients=7, min_available_clients=7, min_evaluate_clients=7), 
                           config=flwr.server.ServerConfig(num_rounds=args.round)
    )
    pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedAvg.csv", index=False)
    pd.DataFrame(history.losses_centralized).to_csv("./Result/FedAvg_loss.csv", index=False)