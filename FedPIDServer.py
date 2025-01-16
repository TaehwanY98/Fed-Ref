from typing import Dict, List, Optional, Tuple
from logging import WARNING
from flwr.common import (
    FitIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr
from flwr.common.logger import log
from flwr.server.strategy.fedavg import aggregate, aggregate_inplace
import torch
from torch.utils.data import DataLoader, random_split
from torch import save
from train import valid , make_model_folder, CustomFocalDiceLoss
import warnings
from utils.parser import Federatedparser
from utils.CustomDataset import Fets2022
from Network.Unet import *
from Network.Loss import *
import getpass
import sys
user = getpass.getuser()
sys.path.append(f"/home/{user}/MICCAI/Network")
from client import seeding
from functools import reduce

import pandas as pd

args = Federatedparser()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
net.to(DEVICE)

cost_list = []
dataset = Fets2022(args.client_dir)
validLoader = DataLoader(dataset, args.batch_size, False, collate_fn=lambda x: x)

def ndarrays_to_arrays(param):
    return [v.flatten() for v in param ]
    
class FedPID(flwr.server.strategy.FedAvg):
    def __init__(self, *, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        
    def aggregate_fit(self, server_round, results, failures):
        
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            if server_round < 3:
                # Convert results
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results
                ]
                aggregated_ndarrays = aggregate(weights_results)
                set_parameters(net, aggregated_ndarrays)
                
                cost = [res.metrics['loss'] for _, res in results]
                cost_list.append(cost)
            else:
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results
                ]
                aggregated_ndarrays = aggregate(weights_results)
                set_parameters(net, aggregated_ndarrays)
                cost = [res.metrics['loss'] for _, res in results]
                cost_list.append(cost)
                kj = [past - pres for past, pres in zip(cost_list[-2],cost_list[-1])]
                K =  sum(kj)
                mj = [past / pres for past, pres in zip(cost_list[0],cost_list[-1])]
                I = sum(mj)
                sj = [fit_res.num_examples for _, fit_res in results]
                S = sum(sj)
                parameters = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
                cgenerators = PID_controller(sj, S, kj, K, mj, I, parameters, len(cost))
                W = reduce(lambda x, y: [xc.ravel()+yc.ravel() for xc, yc in zip(x, y)], cgenerators)
                set_parameters(net, W)
                 
        parameters_aggregated = ndarrays_to_parameters([param.cpu().detach().numpy() for param in net.parameters()])
        
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

def PID_controller(sj, S, kj, K, mj, I, parameters, cnumber=None):
    # J = range(cnumber)
    return ([(0.45*si/S+0.45*ki/K+0.1*mi/I)*p for p in param] for si, ki, mi, param in zip(sj, kj, mj, parameters))
    
    
    
def set_parameters(net, new_parameters):
    for old, new in zip(net.parameters(), new_parameters):
        shape = old.data.size()
        old.data = torch.Tensor(new).view(shape).to(DEVICE)

def fl_save(server_round:int, parameters: flwr.common.NDArrays, config:Dict[str, flwr.common.Scalar], validF=valid)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
    save(net.state_dict(), f"./Models/{args.version}/net.pt")
    print("model is saved")
    return 0, {}

def fl_evaluate(server_round:int, parameters: flwr.common.NDArrays, config:Dict[str, flwr.common.Scalar], validF=valid)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
    set_parameters(net, parameters)
    history=validF(net, validLoader, 0, CustomFocalDiceLoss().to(DEVICE), DEVICE)
    save(net.state_dict(), f"./Models/{args.version}/net.pt")
    return history['loss'], {key:value for key, value in history.items() if key != "loss" }

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    make_model_folder(f"./Models/{args.version}")
    seeding(args)
    
    history = flwr.server.start_server(
        server_address='[::]:8084',strategy=FedPID(evaluate_fn=fl_evaluate, inplace=False, min_fit_clients=7, min_available_clients=7, min_evaluate_clients=7), 
                           config=flwr.server.ServerConfig(num_rounds=args.round)
    )
    pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedPID.csv", index=False)
    pd.DataFrame(history.losses_centralized).to_csv("./Result/FedPID_loss.csv", index=False)