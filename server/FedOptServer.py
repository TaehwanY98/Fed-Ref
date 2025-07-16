from typing import Dict, List, Optional, Tuple
import flwr
import torch
from torch import save
from utils.DriveTrain import valid as validDrive
from utils.TumorTrain import valid 
from utils.octTrain import valid as octValid
from utils.MNISTTrain import valid as MNISTValid
from utils.CIFAR10Train import valid as CIFAR10Valid
from torch import nn
import pandas as pd
import os

from flwr.common import (
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    FitIns
)

from typing import Callable, Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedOpt(flwr.server.strategy.FedOpt):
    def __init__(self, net, lossf, validLoader, args, fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.0,
        beta_2: float = 0.0,
        tau: float = 1e-9,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=ndarrays_to_parameters(initial_parameters),
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_t: Optional[NDArrays] = None
        self.v_t: Optional[NDArrays] = None
        self.net = net
        self.lossf = lossf
        self.args = args
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
            old_historyframe = pd.read_csv(os.path.join(self.args.result_path, self.args.mode, f'{self.args.mode}_{self.args.type}.csv'))
            historyframe = pd.DataFrame({k:[v] for k, v in history.items()})
            newframe=pd.concat([old_historyframe, historyframe])
            newframe.to_csv(os.path.join(self.args.result_path, self.args.mode, f'{self.args.mode}_{self.args.type}.csv'), index=False)
        else:
            pd.DataFrame({k:[v] for k, v in history.items()}).to_csv(os.path.join(self.args.result_path, self.args.mode, f'{self.args.mode}_{self.args.type}.csv'), index=False)
        save(self.net.state_dict(), f"./Models/{self.args.version}/net.pt")
        return history['loss'], {key:value for key, value in history.items() if key != "loss" }
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of training.

        Sends the proximal factor mu to the clients
        """
        # Get the standard client/config pairs from the FedAvg super-class
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Return client/config pairs with the proximal factor mu added
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "tau": self.tau},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]
    
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def set_parameters(net, parameters):
    for old, new in zip(net.parameters(), parameters):
        old.data = torch.Tensor(new).to(DEVICE)