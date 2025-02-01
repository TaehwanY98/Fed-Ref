from typing import Dict, List, Optional, Tuple
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr
import warnings
from flwr.common.logger import log
from flwr.server.strategy.fedavg import aggregate, aggregate_inplace
import torch
from torch.utils.data import DataLoader
from torch import save
from utils.train import valid , make_model_folder, set_seeds, CustomFocalDiceLoss
from utils.octTrain import valid as octValid
from utils.parser import Federatedparser
from utils.CustomDataset import Fets2022, BRATS, OCTDL
from utils.hsic import THSIC, HSIC, customHSIC
from Network.Unet import *
from Network.Loss import *
from Network.Resnet import *
from logging import WARNING
import pandas as pd
from functools import reduce
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hsic = customHSIC

def ndarrays_to_arrays(param):
    return [v.flatten() for v in param]

def CKA(X, Y, hsic, total_num=10, smooth=1e-7):
    result = []
    # for x_, y_ in zip(x,y):
    hsic_xy = hsic(X, Y, total_num)
    hsic_xx = hsic(X, X, total_num)
    hsic_yy = hsic(Y, Y, total_num)
    for xy, xx, yy in zip(hsic_xy, hsic_xx, hsic_yy):
        result.append(1-(xy / (np.sqrt(xx*yy))))
    return result
    

class FedLWR(flwr.server.strategy.FedAvg):
    def __init__(self, net:nn.Module, dataset, validLoader, args,fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.net = net
        self.dataset = dataset
        self.validLoader = validLoader
        self.args = args
        self.evaluate_fn = self.evaluate_fn
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
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            only_weights_results = [
                parameters_to_ndarrays(fit_res.parameters)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)
            # aggregated_ndarrays = aggregate(weights_results)
            Del_k = [CKA(ndarrays_to_arrays(w), ndarrays_to_arrays(aggregated_ndarrays), hsic, len(only_weights_results)) for w in only_weights_results]
            # sumDel_k = reduce(np.add, Del_k)
            # P_k = [delta/sumdel_k for delta, sumdel_k in zip(Del_k, sumDel_k)]
            
            # P_k = [list(map(lambda pw : pw[0].flatten(), zip(pk, we))) for pk, we in zip (P_k, zip(*weights_results))]
            weights_results = [
                (ndarrays_to_arrays(parameters_to_ndarrays(fit_res.parameters)), p_k)
                for (_, fit_res), p_k in zip(results, Del_k)
            ]
            
            num_examples_total = reduce(lambda x, y : [np.add(x1.sum(), y1.sum()) for x1, y1 in zip(x,y)], [num_examples for (_, num_examples) in weights_results])
            weighted_weights = [
                [layer * pk.sum() / total.sum() if not np.isnan(layer*pk.sum()/total.sum()).any() and not np.isinf(layer*pk.sum()/total.sum()).any() else layer/len(weights_results) for layer, pk, total in zip(weights, num_examples, num_examples_total)] for weights, num_examples in weights_results
            ]
            weights_prime = [
                reduce(np.add, layer_updates)
                for layer_updates in zip(*weighted_weights)
                ]
           
        parameters_aggregated = ndarrays_to_parameters(flatten_to_ndarray(self.net, weights_prime))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self, server_round: int, parameters)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
        parameters = parameters_to_ndarrays(parameters)
        lossf = CustomFocalDiceLoss() if not self.args.type=="octdl" else nn.BCEWithLogitsLoss(self.dataset.label_weight)
        validF= valid if not self.args.type=="octdl" else octValid
        set_parameters(self.net, parameters)
        history=validF(self.net, self.validLoader, 0, lossf.to(DEVICE), DEVICE, True)
        make_dir(self.args.result_path)
        make_dir(os.path.join(self.args.result_path, self.args.mode))
        if server_round != 0:
            old_historyframe = pd.read_csv(os.path.join(self.args.result_path, self.args.mode, f'FedLWR_{self.args.type}.csv'))
            historyframe = pd.DataFrame({k:[v] for k, v in history.items()})
            newframe=pd.concat([old_historyframe, historyframe])
            newframe.to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedLWR_{self.args.type}.csv'), index=False)
        else:
            pd.DataFrame({k:[v] for k, v in history.items()}).to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedLWR_{self.args.type}.csv'), index=False)
        save(self.net.state_dict(), f"./Models/{self.args.version}/net.pt")
        return history['loss'], {key:value for key, value in history.items() if key != "loss" }
        
        # return super().aggregate_fit(server_round, results, failures)
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
def set_parameters(net, new_parameters):
    for old, new in zip(net.parameters(), new_parameters):
        shape = old.data.size()
        old.data = torch.Tensor(new).view(shape).to(DEVICE)

def flatten_to_ndarray(net, flatten):
    result = []
    for old, new in zip(net.parameters(), flatten):
        shape = old.data.size()
        result.append(new.reshape(shape))
    return result


if __name__=="__main__":
    warnings.filterwarnings("ignore")
    args = Federatedparser()
    set_seeds(args)
    make_model_folder(f"./Models/{args.version}")
    
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
    
    history = flwr.server.start_server(
        server_address='[::]:8084',strategy=FedLWR(net, dataset, validLoader, args,inplace=False, min_fit_clients=7, min_available_clients=7, min_evaluate_clients=7), 
                           config=flwr.server.ServerConfig(num_rounds=args.round)
    )
    if args.type == 'fets':
        pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedLWR_fets.csv", index=False)
        pd.DataFrame(history.losses_centralized).to_csv("./Result/FedLWR_loss_fets.csv", index=False)
    elif args.type == "brats":
        pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedLWR_BRATS.csv", index=False)
        pd.DataFrame(history.losses_centralized).to_csv("./Result/FedLWR_loss_BRATS.csv", index=False)
    elif args.type == "octdl":
        pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedRef_OCTDL.csv", index=False)
        pd.DataFrame(history.losses_centralized).to_csv("./Result/FedRef_loss_OCTDL.csv", index=False)