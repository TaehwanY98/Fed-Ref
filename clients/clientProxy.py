import flwr 
import torch
import numpy as np
import random
import copy
# from Network.pytorch3dunet.unet3d.losses import BCEDiceLoss
from flwr.common import (
    Code,
    Context,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class CustomNumpyClient(flwr.client.NumPyClient):
    context: Context
    def __init__(self, net, train_loader, length,epoch, lossf, optimizer, DEVICE, args, trainF=lambda x: x, validF=lambda x: x):
        super().__init__()
        self.net = net
        self.train_loader = train_loader
        self.epoch = epoch
        self.lossf = lossf
        self.optim = optimizer
        self.DEVICE=DEVICE
        self.train = trainF
        self.valid = validF
        self.args = args
        self.length = length
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
    
    def fit(self, parameters, config={}):
        self.set_parameters(parameters)
        
        # 서버에서 'proximal_mu' 값이 전달되지 않았을 때를 대비한 기본값(0.0) 설정
        proximal_mu = config.get("proximal_mu", 0.0)
        
        # 이전 전역 모델(Global Model)의 파라미터들을 PyTorch Tensor 리스트로 복사하여 GPU/CPU에 고정
        # 미분이 필요 없으므로 clone().detach()를 사용합니다.
        global_params = [
            torch.tensor(g, dtype=p.dtype).to(self.DEVICE).clone().detach() 
            for p, g in zip(self.net.parameters(), parameters)
        ]

        # PyTorch의 자동 미분 그래프가 유지되는 동적 손실 함수 정의
        def proxy_lossf(outputs, targets):
            # 1. 기본 로컬 손실 계산 (예: CrossEntropy, BCE, Dice 등)
            base_loss = self.lossf(outputs, targets)
            
            if proximal_mu == 0.0:
                return base_loss
                
            # 2. Proximal Term 계산 (L2 Norm의 제곱)
            proximal_term = 0.0
            for local_p, global_p in zip(self.net.parameters(), global_params):
                # (로컬 가중치 - 전역 가중치)의 제곱합 누적
                proximal_term += torch.sum((local_p - global_p) ** 2)
                
            # 3. 최종 손실 반환
            return base_loss + (proximal_mu / 2.0) * proximal_term

        # 안전하게 정의된 proxy_lossf를 로컬 학습 함수에 전달
        self.train(self.net, self.train_loader, None, self.epoch, proxy_lossf, self.optim, self.DEVICE, None)
        
        return self.get_parameters(config={}), self.length, {}

