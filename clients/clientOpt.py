import flwr 
import torch
import numpy as np
import random

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
    def __init__(self, net, train_loader, length, epoch, lossf, optimizer, DEVICE, args, trainF=lambda x: x, validF=lambda x: x):
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
        """서버 최적화(FedOpt) 및 클라이언트 내 근사화 최적화(FedProx)를 동시 지원하는 로컬 학습을 수행합니다."""
        # 1. 서버에서 내려온 최신 글로벌 가중치 설정
        self.set_parameters(parameters)
        
        # 2. 서버 config로부터 proximal_mu(FedProx 하이퍼파라미터) 획득 (기본값 0.0)
        proximal_mu = config.get("proximal_mu", 0.0)
        
        # 3. 글로벌 파라미터를 PyTorch 텐서로 변환하여 고정 (미분 불필요)
        global_params = [
            torch.tensor(g, dtype=p.dtype).to(self.DEVICE).clone().detach() 
            for p, g in zip(self.net.parameters(), parameters)
        ]

        # 4. PyTorch Autograd(자동미분) 그래프를 유지하는 동적 Proximal 손실 함수 정의
        def proxy_lossf(outputs, targets):
            # 기본 태스크 손실 계산 (예: BCE, Dice 등)
            base_loss = self.lossf(outputs, targets)
            
            if proximal_mu == 0.0:
                return base_loss
                
            # Proximal Term 계산 (L2 Norm의 제곱을 PyTorch 연산으로만 수행)
            proximal_term = 0.0
            for local_p, global_p in zip(self.net.parameters(), global_params):
                proximal_term += torch.sum((local_p - global_p) ** 2)
                
            return base_loss + (proximal_mu / 2.0) * proximal_term

        # 5. 주입된 trainF 함수를 통해 로컬 학습 수행 (전달된 proxy_lossf 사용)
        self.train(self.net, self.train_loader, None, self.epoch, proxy_lossf, self.optim, self.DEVICE, None)
        
        # 6. 결과 반환: (업데이트된 파라미터, 데이터 샘플 수, 결과 딕셔너리)
        return self.get_parameters(config={}), self.length, {}

