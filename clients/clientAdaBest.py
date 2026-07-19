from typing import Dict, List, Optional, Tuple
import flwr
import torch
import numpy as np

# Flower의 라운드 간 클라이언트 인스턴스 재생성으로 인한 h_local 초기화 방지를 위한 메모리 dictionary
GLOBAL_CLIENT_STATES = {}

class CustomNumpyClient(flwr.client.NumPyClient):
    def __init__(self, cid, net, train_loader, length, epoch, lossf, optimizer, DEVICE, args, trainF=lambda *args: None, validF=lambda *args: None):
        super().__init__()
        self.cid = str(cid)  # 각 클라이언트를 식별할 고유 ID (예: "0", "1" 등)
        self.net = net
        self.train_loader = train_loader
        self.epoch = epoch
        self.lossf = lossf
        self.optim = optimizer
        self.DEVICE = DEVICE
        self.train = trainF
        self.valid = validF
        self.args = args
        self.length = length
        
        # [AdaBest 보완] 전역 상태 딕셔너리에서 이 클라이언트의 과거 h_local이 있는지 확인하고 로드
        if self.cid not in GLOBAL_CLIENT_STATES:
            GLOBAL_CLIENT_STATES[self.cid] = [
                torch.zeros_like(p, device=self.DEVICE, requires_grad=False) 
                for p in self.net.parameters()
            ]
        self.h_local = GLOBAL_CLIENT_STATES[self.cid]

    def set_parameters(self, parameters):
        """서버로부터 받은 글로벌 가중치를 인플레이스 복사로 안전하게 설정합니다."""
        for old, new in zip(self.net.parameters(), parameters):
            old.data.copy_(torch.tensor(new, dtype=old.dtype).to(self.DEVICE))

    def get_parameters(self, config={}):
        """현재 가중치를 복사하여 안전하게 NumPy 배열 리스트로 반환합니다."""
        return [val.detach().cpu().numpy() for val in self.net.parameters()]

    def fit(self, parameters, config={}):
        """AdaBest 알고리즘의 Client-side Drift Correction을 반영하여 로컬 학습을 수행합니다."""
        # 1. 서버의 최신 글로벌 가중치 설정
        self.set_parameters(parameters)
        
        # 2. 서버 환경설정(config)에서 하이퍼파라미터 로드
        alpha = config['alpha']
        beta = config['beta'] 
        
        # 3. 글로벌 파라미터 백업 (미분 그래프 완전 차단)
        global_params = [
            torch.tensor(g, dtype=p.dtype, device=self.DEVICE).requires_grad_(False)
            for p, g in zip(self.net.parameters(), parameters)
        ]

        # 4. AdaBest 동적 손실 함수 (Adaptive Bias Correction Loss) 정의
        def adabest_lossf(outputs, targets):
            base_loss = self.lossf(outputs, targets)
            
            proximal_term = 0.0
            linear_drift_term = 0.0
            
            for local_p, global_p, h_p in zip(self.net.parameters(), global_params, self.h_local):
                # 가중치 편차 계산 (W_local - W_global)
                param_diff = local_p - global_p
                
                # 정규화 항 및 바이어스 선형 보정 항 적산
                proximal_term += torch.sum(param_diff ** 2)
                linear_drift_term += torch.sum(param_diff * h_p)
                
            # AdaBest 논문 수식 반영: Base Loss + (alpha/2)*Proximal - Linear_Drift
            return base_loss + (alpha / 2.0) * proximal_term - linear_drift_term

        # 5. 주입된 로컬 학습 함수 구동 (adabest_lossf 전달)
        self.train(self.net, self.train_loader, None, self.epoch, adabest_lossf, self.optim, self.DEVICE, None)
        
        # 6. [AdaBest 핵심] 다음 라운드 영속성을 위해 전역 저장소의 h_local 변수 갱신
        with torch.no_grad():
            for h_p, local_p, global_p in zip(self.h_local, self.net.parameters(), global_params):
                drift = local_p.data - global_p.data
                # 기존 객체의 데이터를 인플레이스로 변경하여 주소 유지 및 데이터 갱신
                h_p.copy_(h_p - beta * alpha * drift)
        
        # 7. 갱신된 가중치 저장소 동기화 재확인 후 반환
        GLOBAL_CLIENT_STATES[self.cid] = self.h_local
        
        return self.get_parameters(config={}), self.length, {}