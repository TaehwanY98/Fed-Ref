from typing import Dict, List, Optional, Tuple
import flwr
import torch
from torch import save
from utils.CelebaTrain import valid as celebaValid
from utils.FetsTrain import valid as fetsValid
from utils.Cinic10Train import valid as cinicValid
from utils.FEMNISTTrain import valid as FEMNISTValid
from utils.ShakespeareTrain import valid as shakespeareValid
from utils.OfficeTrain import valid as officeValid
from torch import nn
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
        if self.args.type =="fets":
            validF= fetsValid 
        elif self.args.type=="femnist":
            validF = FEMNISTValid
        elif self.args.type == "cinic10":
            validF = cinicValid
        elif self.args.type == "shakespeare":
            validF = shakespeareValid
        elif self.args.type == "office":
            validF = officeValid
        elif self.args.type == "celeba":
            validF = celebaValid
        set_parameters(self.net, parameters)
        history=validF(self.net, self.validLoader, 0, self.lossf.to(DEVICE), DEVICE, True)
        make_dir(self.args.result_path)
        make_dir(os.path.join(self.args.result_path, self.args.mode))
        if server_round != 0:
            old_historyframe = pd.read_csv(os.path.join(self.args.result_path, self.args.mode, f'{self.args.mode}_{self.args.type}.csv'))
            historyframe = pd.DataFrame({k:[v] for k, v in history.items()})
            newframe=pd.concat([old_historyframe, historyframe])
            newframe.to_csv(os.path.join(self.args.result_path, self.args.mode, f'{self.args.mode}_{self.args.type}.csv'), index=False)
        else:
            pd.DataFrame({k:[v] for k, v in history.items()}).to_csv(os.path.join(self.args.result_path, self.args.mode, f'{self.args.mode}_{self.args.type}.csv'), index=False)
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
