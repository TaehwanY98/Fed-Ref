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
class AdaBest(flwr.server.strategy.FedAvg):
    def __init__(self, net, lossf, validLoader, args, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None):
        # 복사본 기반 연산을 안전하게 수행하기 위해 inplace=False 강제 설정
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=False)
        self.net = net
        self.args = args
        self.lossf = lossf
        self.validLoader = validLoader
        self.evaluate_fn = self.evaluate_fn
        self.initial_global_model= copy.deepcopy(net)
        # [AdaBest] 직전 라운드의 글로벌 가중치를 추적하기 위한 변수
        self.prev_global_parameters: Optional[List[np.ndarray]] = None
        # [AdaBest] 클라이언트 그라디언트 제어용 노름 임계치 초깃값 및 모멘텀 계수 beta
        self.clip_norm_threshold = 1.0
        self.alpha = 0.1
        self.beta = 0.9
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        self.local_random = random.Random(self.args.seed)

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
    def configure_fit(
        self, server_round: int, parameters: flwr.common.Parameters, client_manager: flwr.server.client_manager.ClientManager
    ) -> List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitIns]]:
        """다음 라운드 학습 시 클라이언트들에게 글로벌 가중치와 노름 제약치(clip_norm)를 함께 배포합니다."""
        # 1. 기존에 별도의 on_fit_config_fn이 구현되어 있다면 가져오고, 없으면 빈 딕셔너리로 시작
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # 2. [AdaBest 핵심] 서버가 계산한 동적 노름 임계값을 config에 주입하여 전달
        config["clip_norm"] = self.clip_norm_threshold
        config["alpha"] = self.alpha
        config["beta"] = self.beta
        
        fit_ins = flwr.common.FitIns(parameters, config)
        
        # 3. 사용 가능한 클라이언트 샘플링
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        pprint(vars(self.args))
        if not results or (not self.accept_failures and failures):
            return None, {}

        # 1. 첫 라운드일 경우 기준이 되는 이전 글로벌 매개변수를 초기화
        if self.prev_global_parameters is None:
            first_client_param = parameters_to_ndarrays(results[0][1].parameters)
            self.prev_global_parameters = [np.zeros_like(layer) for layer in first_client_param]

        # 2. 클라이언트 결과 파싱
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # 3. [AdaBest 핵심] 클라이언트들이 보낸 업데이트 변화량(Drift)의 L2 노름 추적
        total_drift_norm = 0.0
        for client_arrays, _ in weights_results:
            # 레이어별 Drift Vector (Client weight - Prev Global weight) 계산
            layer_drifts = [c - p for c, p in zip(client_arrays, self.prev_global_parameters)]
            # 전체 가중치 평탄화 후 L2 노름 연산
            drift_norm = np.sqrt(sum(np.sum(np.square(d)) for d in layer_drifts))
            total_drift_norm += drift_norm

        avg_drift_norm = total_drift_norm / max(len(results), 1)

        # 4. 부모 클래스의 FedAvg 결합 알고리즘을 활용하여 새로운 글로벌 가중치 계산
        aggregated_parameters = self.warm_up(results=results)
        self.initial_global_update()
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        if aggregated_parameters is not None:
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            
            # 5. [AdaBest 핵심] 다음 기수 학습에 사용할 노름 임계값 동적 갱신 (지수 이동 평균)
            if server_round > 1:
                self.clip_norm_threshold = self.beta * self.clip_norm_threshold + (1 - self.beta) * avg_drift_norm
            
            # 6. 다음 차분 연산을 위해 현재 글로벌 가중치 백업
            self.prev_global_parameters = copy.deepcopy(aggregated_ndarrays)

        return aggregated_parameters, metrics_aggregated

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
            
            self.set_parameters(self.aggregated_net, ndarrays)
            history = validF(self.aggregated_net, self.validLoader, 0, self.lossf.to(self.DEVICE), self.DEVICE, True)
            
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

    def set_parameters(self, parameters):
        for old, new in zip(self.net.parameters(), parameters):
            old.data.copy_(torch.tensor(new, dtype=old.dtype).to(self.DEVICE))

    def get_parameters(self, net, config={}):
            """현재 클라이언트 모델의 가중치를 NumPy 배열 리스트로 변환하여 반환합니다."""
            # 미분 그래프 추적을 끊고(detach), CPU로 이동 후, 안전하게 numpy 배열로 변환
            return [val.detach().cpu().numpy() for val in net.parameters()]

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)