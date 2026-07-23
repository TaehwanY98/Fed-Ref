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
from pprint import pprint
import copy
from flwr.server.strategy.fedavg import aggregate
class FedProx(flwr.server.strategy.FedProx):
    def __init__(self, net, lossf, validLoader, args, proximal_mu=1,fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(proximal_mu=proximal_mu, fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
        self.net = net
        self.lossf = lossf
        self.args = args
        self.validLoader = validLoader
        self.evaluate_fn = self.evaluate_fn
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        self.initial_global_model= copy.deepcopy(net)


    def warm_up(self, results):
        """웜업 기간 또는 UDP 조건 미충족 시 수행되는 기본 FedAvg 구조"""
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) if not torch.rand(size=1).item() < self.args.degrade else (parameters_to_ndarrays(self.get_parameters(self.initial_global_model)), fit_res.num_examples)
            for _, fit_res in results 
        ]
        aggregated_ndarrays = aggregate(weights_results)
        return aggregated_ndarrays

    def aggregate_fit(self, server_round, results, failures):
        if not results or (not self.accept_failures and failures):
            return None, {}
        
        pprint(vars(self.args))

        aggregated_parameters = self.warm_up(results=results)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        return aggregated_parameters, metrics_aggregated
    
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
        self.set_parameters(parameters)
        history=validF(self.net, self.validLoader, 0, self.lossf.to(self.DEVICE), self.DEVICE, True)
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
    def set_parameters(self, parameters):
        """서버로부터 받은 가중치를 클라이언트 모델에 안전하게 적용합니다."""
        for old, new in zip(self.net.parameters(), parameters):
            # 1. torch.Tensor(new) 대신 소문자 torch.tensor(new) 사용 (타입 추론 및 안전성)
            # 2. old.data = ... 방식은 참조를 깨뜨리므로 .copy_()를 사용하여 인플레이스(In-place) 덮어쓰기 수행
            old.data.copy_(torch.tensor(new, dtype=old.dtype).to(self.DEVICE))

    def get_parameters(self, config={}):
        """현재 클라이언트 모델의 가중치를 NumPy 배열 리스트로 변환하여 반환합니다."""
        # 미분 그래프 추적을 끊고(detach), CPU로 이동 후, 안전하게 numpy 배열로 변환
        return [val.detach().cpu().numpy() for val in self.net.parameters()]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)