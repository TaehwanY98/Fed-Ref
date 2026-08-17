from typing import Dict, List, Optional, Tuple
import flwr
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.strategy.fedavg import aggregate
from torch import nn
import numpy as np
import torch
import copy
import os
import pandas as pd
from logging import WARNING
from utils.FetsTrain import valid as fetsValid
from utils.CelebaTrain import valid as celebaValid
from utils.Cinic10Train import valid as cinicValid
from utils.FEMNISTTrain import valid as FEMNISTValid
from utils.ShakespeareTrain import valid as shakespeareValid
from utils.OfficeTrain import valid as officeValid
from pprint import pprint
import copy
import random

class FedRef(flwr.server.strategy.FedAvg):
    def __init__(self, ref_net: nn.Module, aggregated_net: nn.Module, lossf, validLoader, args, p: int = 2, **kwargs):
        # inplace=False를 강제하여 복사본 기반 안전한 연산 보장
        kwargs["inplace"] = False
        super().__init__(**kwargs)
        
        self.ref_net = ref_net
        self.aggregated_net = aggregated_net
        self.lossf = lossf
        self.validLoader = validLoader
        self.args = args
        self.p = p
        self.initial_global_model= copy.deepcopy(aggregated_net)
        # 과거 p개의 글로벌 모델 파라미터를 담을 윈도우 리스트 (최신 p개 유지)
        self.global_history: List[List[np.ndarray]] = []
        # 직전 라운드의 가중 평균 글로벌 손실값 저장
        self.prev_global_loss: Optional[float] = None
        self.local_random = random.Random(self.args.seed)
        # 최초 reference 파라미터 초기화
        self.theta0_ref: Optional[List[np.ndarray]] = None
        self.theta0_agg: Optional[List[np.ndarray]] = None
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        
    def calculate_reference_model(self) -> List[np.ndarray]:
        """논문 수식 (9), (10)번에 따른 시간 가중 윈도우 기반 참조 모델 생성"""
        rho = len(self.global_history)
        phi = sum(range(1, rho + 1))  # Normalization factor Φ
        
        # 각 레이어별로 초기화
        num_layers = len(self.global_history[0])
        ref_model = [np.zeros_like(layer) for layer in self.global_history[0]]
        
        # 오래된 라운드일수록 i가 작고, 최신일수록 i가 큼 (역순 가중치 적용)
        # 논문 수식: (ρ - i + 1) / Φ * θ_{r+1-i}^g
        for idx, theta_g in enumerate(self.global_history):
            i = idx + 1  # 1부터 rho까지
            weight = (rho - i + 1) / phi
            for l in range(num_layers):
                ref_model[l] += weight * theta_g[l]
        return ref_model
    
    def initial_global_update(self):
        self.initial_global_model = copy.deepcopy(self.aggregated_net)

    def warm_up(self, results):
        """웜업 기간 또는 UDP 조건 미충족 시 수행되는 기본 FedAvg 구조"""
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) if not self.local_random.random() < self.args.degrade else (self.get_parameters(self.initial_global_model), fit_res.num_examples)
            for _, fit_res in results 
        ]
        aggregated_ndarrays = aggregate(weights_results)
        return aggregated_ndarrays

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]], failures: List[BaseException]
    ) -> Tuple[Optional[flwr.common.Parameters], Dict[str, flwr.common.Scalar]]:
        pprint(vars(self.args))
        if not results or (not self.accept_failures and failures):
            return None, {}

        # 1. 클라이언트 가중치(W_k) 및 가중 평균 손실값(F^g) 계산
        agg_examples = [res.num_examples for _, res in results]
        total_examples = sum(agg_examples)
        client_weights = [n / total_examples for n in agg_examples]
        
        # 클라이언트가 전송한 'loss' 추출 후 가중 평균 계산
        try:
            client_losses = [res.metrics["loss"] for _, res in results]
            current_global_loss = sum(w * loss for w, loss in zip(client_weights, client_losses))
        except KeyError:
            log(WARNING, "Client did not return 'loss' in metrics. FedRef requires loss tracking.")
            current_global_loss = 0.0

        # 2. 기본 FedAvg 결합 수행 (θ_{r+1}^g)
        aggregated_ndarrays = self.warm_up(results)
        self.initial_global_update()
        
        # 3. 웜업 페이즈 처리 (History 윈도우 p 채우기)
        if server_round <= self.p:
            self.global_history.append(copy.deepcopy(aggregated_ndarrays))
            self.prev_global_loss = current_global_loss
            
            # 다음 라운드를 위한 theta0 저장
            self.theta0_agg = copy.deepcopy(aggregated_ndarrays)
            return ndarrays_to_parameters(aggregated_ndarrays), {}

        # 4. UDP Detection (Drift 감지 조건 체크)
        delta_F = 0.0
        if self.prev_global_loss is not None:
            delta_F = current_global_loss - self.prev_global_loss

        # 조건 만족하지 않으면 Fine-tuning 없이 통과 (수식 조건: ΔF^g > δ)
        if delta_F <= self.args.delta:
            # 윈도우 업데이트 (가장 오래된 것 제거 후 최신 것 추가)
            self.global_history.pop(0)
            self.global_history.append(copy.deepcopy(aggregated_ndarrays))
            self.prev_global_loss = current_global_loss
            self.theta0_agg = copy.deepcopy(aggregated_ndarrays)
            return ndarrays_to_parameters(aggregated_ndarrays), {}

        # 5. UDP 조건 충족 시 Bayesian Fine-Tuning 진행
        # 가중 이동 평균 기반 참조 모델(θ^ref) 도출
        ref_ndarrays = self.calculate_reference_model()
        
        if self.theta0_ref is None:
            self.theta0_ref = copy.deepcopy(ref_ndarrays)

        # 수식 파라미터 준비: Δθ_r^g 및 Δθ_r^ref 계산
        # Δθ_r^g = θ_{r+1}^g - θ_r^g
        delta_theta_g = [g - t0 for g, t0 in zip(aggregated_ndarrays, self.theta0_agg)]
        # Δθ_r^ref = θ_{r+1}^ref - θ_r^ref
        delta_theta_ref = [r - t0 for r, t0 in zip(ref_ndarrays, self.theta0_ref)]

        # 6. Server-side Bayesian Fine-Tuning (Eq. 14) 실행
        # F_ref = Likelihood(데이터평균) + λ^g * ||Δθ^g - Δθ^ref||^2
        # 이를 θ^g에 대해 미분하여 1 step Gradient Descent 보정 적용
        corrected_ndarrays = []
        for g, d_g, d_ref in zip(aggregated_ndarrays, delta_theta_g, delta_theta_ref):
            # Regularization 항의 그래디언트: 2 * λ^g * (Δθ^g - Δθ^ref)
            reg_gradient = 2 * self.args.lda * (d_g - d_ref)
            # 1 step Gradient Descent 보정
            corrected_layer = g - self.args.lr * reg_gradient
            corrected_ndarrays.append(corrected_layer)

        # 7. 차기 라운드를 위한 상태 변수 갱신
        self.global_history.pop(0)
        self.global_history.append(copy.deepcopy(corrected_ndarrays))
        self.prev_global_loss = current_global_loss
        self.theta0_agg = copy.deepcopy(corrected_ndarrays)
        self.theta0_ref = copy.deepcopy(ref_ndarrays)

        # 가중치 메트릭 결합 (있을 경우)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(corrected_ndarrays), metrics_aggregated

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
        os.makedirs(os.path.join(self.args.result_path, self.args.mode, f"degrade{self.args.degrade}", self.args.clientMode, str(self.args.seed)), exist_ok=True)
        csv_path = os.path.join(self.args.result_path, self.args.mode, str(self.args.seed), f"degrade{self.args.degrade}", self.args.clientMode, str(self.args.seed), f'{self.args.mode}_{self.args.type}_lda{self.args.lda*10}_p{self.args.prime}.csv')
        
        historyframe = pd.DataFrame({k: [v] for k, v in history.items()})
        if server_round != 0 and os.path.exists(csv_path):
            old_historyframe = pd.read_csv(csv_path)
            newframe = pd.concat([old_historyframe, historyframe])
            newframe.to_csv(csv_path, index=False)
        else:
            historyframe.to_csv(csv_path, index=False)
            
        return history['loss'], {key: value for key, value in history.items() if key != "loss"}
    def set_parameters(self, net, parameters):
        """서버로부터 받은 가중치를 클라이언트 모델에 안전하게 적용합니다."""
        for old, new in zip(net.parameters(), parameters):
            # 1. torch.Tensor(new) 대신 소문자 torch.tensor(new) 사용 (타입 추론 및 안전성)
            # 2. old.data = ... 방식은 참조를 깨뜨리므로 .copy_()를 사용하여 인플레이스(In-place) 덮어쓰기 수행
            old.data.copy_(torch.tensor(new, dtype=old.dtype).to(self.DEVICE))

    def get_parameters(self, net, config={}):
        """현재 클라이언트 모델의 가중치를 NumPy 배열 리스트로 변환하여 반환합니다."""
        # 미분 그래프 추적을 끊고(detach), CPU로 이동 후, 안전하게 numpy 배열로 변환
        return [val.detach().cpu().numpy() for val in net.parameters()]