from typing import Dict, List, Optional, Tuple
import flwr
import torch
from torch import save
import pandas as pd
import os
import numpy as np
import copy
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from pprint import pprint
from utils.CelebaTrain import valid as celebaValid
from utils.FetsTrain import valid as fetsValid
from utils.Cinic10Train import valid as cinicValid
from utils.FEMNISTTrain import valid as FEMNISTValid
from utils.ShakespeareTrain import valid as shakespeareValid
from utils.OfficeTrain import valid as officeValid
import copy
from flwr.server.strategy.fedavg import aggregate
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from flwr.common import ndarrays_to_parameters
class A_FedPD(flwr.server.strategy.FedAvg):
    def __init__(self, net, lossf, validLoader, args, fraction_fit=1, fraction_evaluate=1, min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2, **kwargs):
        kwargs["inplace"] = False  # 안전한 갱신을 위해 복사본 모드 강제
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, **kwargs)
        
        self.net = net
        self.args = args
        self.lossf = lossf
        self.validLoader = validLoader
        self.local_random = random.Random(self.args.seed)
        # [A-FedPD] 가상 듀얼 업데이트 및 전역 드리프트 제어를 위한 하이퍼파라미터
        self.mu_param = getattr(args, 'mu_param', 0.1)  # 클라이언트 정규화와 매핑되는 글로벌 mu
        self.initial_global_model= copy.deepcopy(net)
    def initial_global_update(self):
                self.initial_global_model = copy.deepcopy(self.net)

    def warm_up(self, results):
        """웜업 기간 또는 UDP 조건 미충족 시 수행되는 기본 FedAvg 구조"""
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) if not self.local_random.random() < self.args.degrade else (self.get_parameters(self.initial_global_model), fit_res.num_examples)
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
        
        fit_ins = flwr.common.FitIns(parameters, config)
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results or (not self.accept_failures and failures):
            return None, {}
        
        pprint(vars(self.args))

        aggregated_parameters = self.warm_up(results=results)
        self.initial_global_update()
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        return ndarrays_to_parameters(aggregated_parameters), metrics_aggregated

    def evaluate(self, server_round: int, parameters) -> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
            # 기존 evaluate 로직 유지 (타입별 밸리데이션 및 csv 저장)
            ndarrays = parameters_to_ndarrays(parameters)
            
            # 패키지 내 함수 세팅 (사용자 정의 함수 구현체 필수)
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
            
            self.set_parameters(self.net, ndarrays)
            history = validF(self.net, self.validLoader, 0, self.lossf.to(self.DEVICE), self.DEVICE, True)
            
            # 파일 저장 경로 처리
            os.makedirs(os.path.join(self.args.result_path, self.args.mode, f"degrade{self.args.degrade}"), exist_ok=True)
            csv_path = os.path.join(self.args.result_path, self.args.mode, f"degrade{self.args.degrade}", f'{self.args.mode}_{self.args.type}_lda{self.args.lda*10}_p{self.args.prime}.csv')
            
            historyframe = pd.DataFrame({k: [v] for k, v in history.items()})
            if server_round != 0 and os.path.exists(csv_path):
                old_historyframe = pd.read_csv(csv_path)
                newframe = pd.concat([old_historyframe, historyframe])
                newframe.to_csv(csv_path, index=False)
            else:
                historyframe.to_csv(csv_path, index=False)
                
            return history['loss'], {key: value for key, value in history.items() if key != "loss"}

    def set_parameters(self, net , parameters):
        for old, new in zip(net.parameters(), parameters):
            old.data.copy_(torch.tensor(new, dtype=old.dtype).to(DEVICE))
    def get_parameters(self, net, config={}):
            """현재 클라이언트 모델의 가중치를 NumPy 배열 리스트로 변환하여 반환합니다."""
            # 미분 그래프 추적을 끊고(detach), CPU로 이동 후, 안전하게 numpy 배열로 변환
            return [val.detach().cpu().numpy() for val in net.parameters()]
def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)