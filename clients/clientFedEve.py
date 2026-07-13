from typing import Dict, List, Optional, Tuple
import flwr
import torch
import numpy as np

# Flower의 라운드 전환 시 인스턴스 초기화 문제를 해결하기 위한 외부 상태 저장소
GLOBAL_EVE_STATES = {}

class CustomNumpyClient(flwr.client.NumPyClient):
    def __init__(self, cid, net, train_loader, length, epoch, lossf, optimizer, DEVICE, args, trainF=lambda *args: None, validF=lambda *args: None):
        super().__init__()
        self.cid = str(cid)  # 클라이언트 고유 ID 추가
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
        
        # [FedEve 보완] 전역 상태 저장소에서 이 클라이언트의 v_local 히스토리가 있는지 확인
        if self.cid not in GLOBAL_EVE_STATES:
            GLOBAL_EVE_STATES[self.cid] = [
                torch.zeros_like(p, device=self.DEVICE, requires_grad=False) 
                for p in self.net.parameters()
            ]
        self.v_local = GLOBAL_EVE_STATES[self.cid]

    def set_parameters(self, parameters):
        """글로벌 모델의 파라미터를 인플레이스(In-place)로 안전하게 설정합니다."""
        for old, new in zip(self.net.parameters(), parameters):
            old.data.copy_(torch.tensor(new, dtype=old.dtype).to(self.DEVICE))

    def get_parameters(self, config={}):
        """현재 클라이언트 모델의 파라미터를 NumPy 배열 리스트로 안전하게 반환합니다."""
        return [val.detach().cpu().numpy() for val in self.net.parameters()]

    def fit(self, parameters, config={}):
        """FedEve 알고리즘의 변동성 감소(Variance Reduction) 및 로컬 정규화를 적용하여 학습합니다."""
        # 1. 최신 글로벌 파라미터 적용
        self.set_parameters(parameters)
        
        # 2. 하이퍼파라미터 로드
        mu = config.get("mu", 0.01)       # 로컬 정규화 강도
        rho = config.get("rho", 0.9)      # 분산 제어 모멘텀 계수
        
        # 3. 글로벌 파라미터 복사 (미분 그래프 완전 제외 및 디바이스 지정 고정)
        global_params = [
            torch.tensor(g, dtype=p.dtype, device=self.DEVICE).requires_grad_(False)
            for p, g in zip(self.net.parameters(), parameters)
        ]

        # 4. PyTorch Autograd가 정상 추적하는 동적 FedEve 손실 함수 정의
        def fedeve_lossf(outputs, targets):
            base_loss = self.lossf(outputs, targets)
            
            proximal_term = 0.0
            variance_corr_term = 0.0
            
            for local_p, global_p, v_p in zip(self.net.parameters(), global_params, self.v_local):
                # (W_local - W_global) 차이 계산
                diff = local_p - global_p
                
                # ① 로컬-글로벌 괴리를 막는 L2 페널티
                proximal_term += torch.sum(diff ** 2)
                
                # ② 과거 분산 값을 보정하는 선형 수정 항
                variance_corr_term += torch.sum(diff * v_p)
                
            # 최종 손실 함수: Base + Regularization + Variance Correction
            return base_loss + (mu / 2.0) * proximal_term + variance_corr_term

        # 5. 정의된 동적 손실 함수를 주입하여 로컬 학습 진행
        self.train(self.net, self.train_loader, None, self.epoch, fedeve_lossf, self.optim, self.DEVICE, None)
        
        # 6. 학습 종료 후 다음 라운드를 위한 로컬 상태 변수(v_local) 업데이트
        with torch.no_grad():
            for v_p, local_p, global_p in zip(self.v_local, self.net.parameters(), global_params):
                current_drift = local_p.data - global_p.data
                # 기존 텐서 메모리 주소 내에서 인플레이스로 데이터 갱신 (copy_)
                v_p.copy_(rho * v_p + (1.0 - rho) * current_drift)
        
        # 7. 갱신된 텐서 리스트 상태를 전역 저장소에 재동기화
        GLOBAL_EVE_STATES[self.cid] = self.v_local
        
        return self.get_parameters(config={}), self.length, {}