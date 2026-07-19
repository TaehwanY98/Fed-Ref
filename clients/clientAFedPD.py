import flwr 
import types
import torch
import numpy as np

# 클라이언트별 듀얼(λ) 및 모멘텀 변수 영속 유지를 위한 중앙 저장소
GLOBAL_CLIENT_HISTORIES = {}

class CustomNumpyClient(flwr.client.NumPyClient):
    def __init__(self, cid, net, train_loader, length, epoch, lossf, optimizer, DEVICE, args, trainF=lambda x: x, validF=lambda x: x):
        super().__init__()
        
        self.cid = str(cid)
        self.net = net
        self.train_loader = train_loader
        self.epoch = epoch
        self.lossf = lossf
        self.optim = optimizer
        self.DEVICE = DEVICE
        self.args = args
        self.length = length
        self.train = trainF
        self.valid = validF

        # 글로벌 저장소 공간 확보
        if self.cid not in GLOBAL_CLIENT_HISTORIES:
            GLOBAL_CLIENT_HISTORIES[self.cid] = {}

        # 1. args.clientMode 조건 분기
        if args.clientMode == "afedpd":
            # A-FedPD 듀얼 변수(λ) 복원 또는 최초 영행렬 초기화
            if "lambda_local" not in GLOBAL_CLIENT_HISTORIES[self.cid]:
                GLOBAL_CLIENT_HISTORIES[self.cid]["lambda_local"] = [
                    torch.zeros_like(p, device=self.DEVICE, requires_grad=False) for p in self.net.parameters()
                ]
            self.lambda_local = GLOBAL_CLIENT_HISTORIES[self.cid]["lambda_local"]
            
            # 메서드 동적 주입 (앞서 정의한 확장 스크립트 연결 예시)
            self.fit = types.MethodType(afedpd_fit_extension, self)
            
        elif args.clientMode == "fedavg":
            import client as avg
            self.fit = types.MethodType(avg.CustomNumpyClient.fit, self)
        # ... 타 알고리즘 생략 ...

    def fit(self, parameters, config={}):
        """A-FedPD 알고리즘의 Primal-Dual 로컬 최적화 단계"""
        # 1. 서버의 최신 글로벌 가중치 설정
        self.set_parameters(parameters)
        
        # 2. 하이퍼파라미터 로드 (라그랑주 승수 강도 및 패널티 계수)
        mu = config["mu"]  # 증강 라그랑주 분산 패널티 강도 (p)
        
        # 3. 글로벌 파라미터 복사 (미분 완전 제외 및 고정)
        global_params = [
            torch.tensor(g, dtype=p.dtype, device=self.DEVICE).requires_grad_(False)
            for p, g in zip(self.net.parameters(), parameters)
        ]

        # 4. A-FedPD Primal-Dual 증강 라그랑주 손실 함수 정의
        def afedpd_lossf(outputs, targets):
            base_loss = self.lossf(outputs, targets)
            
            lagrangian_term = 0.0
            proximal_term = 0.0
            
            # self.lambda_local은 동적 팩토리 클래스에서 영속 유지됨
            for local_p, global_p, lam_p in zip(self.net.parameters(), global_params, self.lambda_local):
                diff = local_p - global_p
                
                # ① 선형 듀얼 항 (Linear Dual Term): <λ, W_local - W_global>
                lagrangian_term += torch.sum(diff * lam_p)
                # ② 증강 이차 패널티 항 (Augmented Quadratic Penalty): (μ / 2) * ||W_local - W_global||^2
                proximal_term += torch.sum(diff ** 2)
                
            # L(W) = f(W) + <λ, W - W_g> + (μ/2) * ||W - W_g||^2
            return base_loss + lagrangian_term + (mu / 2.0) * proximal_term

        # 5. 정의된 동적 Primal-Dual 손실 함수를 주입하여 로컬 학습 수행 (Primal Update)
        self.train(self.net, self.train_loader, None, self.epoch, afedpd_lossf, self.optim, self.DEVICE, None)
        
        # 6. [A-FedPD 핵심] 학습 종료 후 로컬 듀얼 변수(λ) 업데이트 (Dual Ascent)
        # λ^{t} = λ^{t-1} + μ * (W_local - W_global)
        with torch.no_grad():
            for lam_p, local_p, global_p in zip(self.lambda_local, self.net.parameters(), global_params):
                drift = local_p.data - global_p.data
                lam_p.copy_(lam_p + mu * drift)
                
        return self.get_parameters(config={}), self.length, {}
    
    def set_parameters(self, parameters):
        for old, new in zip(self.net.parameters(), parameters):
            old.data.copy_(torch.tensor(new, dtype=old.dtype).to(self.DEVICE))

    def get_parameters(self, config={}):
        return [val.detach().cpu().numpy() for val in self.net.parameters()]