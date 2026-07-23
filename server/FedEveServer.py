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
import numpy as np
import copy
from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from pprint import pprint
import copy
from flwr.server.strategy.fedavg import aggregate
import random
class FedEve(flwr.server.strategy.FedAvg):
    def __init__(self, net, lossf, validLoader, args, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None):
        # 복사본 기반 연산을 안전하게 수행하기 위해 inplace=False 강제 설정
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=False)
        self.net = net
        self.args = args
        self.lossf = lossf
        self.validLoader = validLoader
        self.evaluate_fn = self.evaluate_fn
        self.local_random = random.Random(self.args.seed)
        
        # [FedEve] Period Drift와 Client Drift를 상쇄하기 위한 서버 제어 변수
        self.v_server: Optional[List[np.ndarray]] = None  # 전역 모멘텀/예측 벡터 (Predict Vector)
        self.prev_global_parameters: Optional[List[np.ndarray]] = None # 직전 라운드 모델
        
        # [FedEve 하이퍼파라미터] 논문 기준 기본 세팅 값
        self.server_lr = getattr(args, 'server_lr', 1.0)  # 서버 스텝 사이즈 (η_s)
        self.momentum_beta = getattr(args, 'momentum_beta', 0.9)  # 모멘텀 계수 (β)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        self.rho_param = getattr(args, "rho_param", 0.9)
        self.mu_param = getattr(args, "mu_param", 0.01)
        self.initial_global_model= copy.deepcopy(net)
    def warm_up(self, results):
        """웜업 기간 또는 UDP 조건 미충족 시 수행되는 기본 FedAvg 구조"""
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) if not self.local_random.random() < self.args.degrade else (parameters_to_ndarrays(self.get_parameters(self.initial_global_model)), fit_res.num_examples)
            for _, fit_res in results 
        ]
        aggregated_ndarrays = aggregate(weights_results)
        return aggregated_ndarrays
    def configure_fit(self, server_round: int, parameters: flwr.common.Parameters, client_manager: flwr.server.client_manager.ClientManager):
        """클라이언트가 로컬 Primal-Dual 연산을 수행할 수 있도록 설정값(mu)을 전달합니다."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
            
        config["mu"] = self.mu_param  # 클라이언트 손실함수에 mu 동적 주입
        config["rho"] = self.mu_param
        
        fit_ins = flwr.common.FitIns(parameters, config)
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        pprint(vars(self.args))
        if not results or (not self.accept_failures and failures):
            return None, {}

        # 1. 첫 라운드일 경우 기준 매개변수 및 예측 벡터 초기화
        if self.prev_global_parameters is None:
            first_client_param = parameters_to_ndarrays(results[0][1].parameters)
            self.prev_global_parameters = copy.deepcopy(first_client_param)
            self.v_server = [np.zeros_like(layer) for layer in first_client_param]

        # 2. 부모 클래스의 FedAvg 결합 알고리즘을 활용하여 1차 관찰 가중치(Observed Consensus) 계산
        aggregated_parameters = self.warm_up(results=results)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        if aggregated_parameters is not None:
            observed_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            
            # 3. [FedEve 핵심: Predict-Observe Framework]
            # 이번 라운드 클라이언트 군집의 총 변화량(Observed Drift Vector) 계산
            # Delta_theta = Theta_observed - Theta_prev
            observed_drift = [obs - prev for obs, prev in zip(observed_ndarrays, self.prev_global_parameters)]
            
            corrected_ndarrays = []
            new_v_server = []
            
            for i in range(len(observed_ndarrays)):
                # 서버 측 예측 모멘텀 벡터 갱신: v^{t} = β * v^{t-1} + (1 - β) * observed_drift
                v_layer = self.momentum_beta * self.v_server[i] + (1 - self.momentum_beta) * observed_drift[i]
                new_v_server.append(v_layer)
                
                # 최종 전역 모델 갱신: Period Drift 보정을 위해 예측 벡터 성분을 조합하여 보정
                # Theta^{t} = Theta^{t-1} + η_s * v^{t}
                corrected_layer = self.prev_global_parameters[i] + self.server_lr * v_layer
                corrected_ndarrays.append(corrected_layer)
                
            # 4. 차기 라운드 추적용 변수 업데이트
            self.v_server = new_v_server
            self.prev_global_parameters = copy.deepcopy(corrected_ndarrays)
            
            # 최종 연산 결과를 Flower 파라미터 타입으로 재변환
            aggregated_parameters = ndarrays_to_parameters(corrected_ndarrays)

        return aggregated_parameters, metrics_aggregated

    def evaluate(self, server_round: int, parameters)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
        # 제공해주신 기존 평가 및 CSV 저장 로직 유지
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
        
        csv_path = os.path.join(self.args.result_path, self.args.mode, f'{self.args.mode}_{self.args.type}.csv')
        historyframe = pd.DataFrame({k:[v] for k, v in history.items()})
        
        if server_round != 0 and os.path.exists(csv_path):
            old_historyframe = pd.read_csv(csv_path)
            newframe = pd.concat([old_historyframe, historyframe])
            newframe.to_csv(csv_path, index=False)
        else:
            historyframe.to_csv(csv_path, index=False)
            
        save(self.net.state_dict(), f"./Models/{self.args.version}/net.pt")
        return history['loss'], {key:value for key, value in history.items() if key != "loss" }

    def set_parameters(self, parameters):
        for old, new in zip(self.net.parameters(), parameters):
            old.data.copy_(torch.tensor(new, dtype=old.dtype).to(self.DEVICE))

    def get_parameters(self, config={}):
        return [val.detach().cpu().numpy() for val in self.net.parameters()]

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)