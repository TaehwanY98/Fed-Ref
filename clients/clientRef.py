import flwr 
import types  # 동적 메서드 바인딩을 위한 빌트인 모듈
import torch
# 사용자 정의 모듈 임포트
import client as avg
import clientProxy as prox
import clientOpt as opt
import clientFedEve as eve
import clientAdaBest as adabest
# 라운드가 바뀌어 클라이언트 인스턴스가 초기화되어도 상태를 기억하기 위한 저장소
GLOBAL_CLIENT_HISTORIES = {}
import flwr 
import types  # 동적 메서드 바인딩을 위한 빌트인 모듈
import torch
import numpy as np

# 사용자 정의 모듈 임포트
import client as avg
import clientProxy as prox
import clientOpt as opt
import clientFedEve as eve
import clientAdaBest as adabest

# 각 클라이언트 고유 히스토리 영속성을 위한 글로벌 딕셔너리
GLOBAL_CLIENT_HISTORIES = {}

class CustomNumpyClient(flwr.client.NumPyClient):
    def __init__(self, cid, net, train_loader, length, epoch, lossf, optimizer, DEVICE, args, trainF=lambda x: x, validF=lambda x: x):
        super().__init__()
        
        # 공통 기본 속성 정의
        self.cid = str(cid)  # 클라이언트 고유 ID (주입 필수)
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

        # 글로벌 상태 저장소에 현재 클라이언트의 고유 공간이 없으면 초기화
        if self.cid not in GLOBAL_CLIENT_HISTORIES:
            GLOBAL_CLIENT_HISTORIES[self.cid] = {}

        # 1. args.clientMode 조건에 따라 소스 클래스 결정
        target_class = None
        
        if args.clientMode == "fedavg":
            target_class = avg.CustomNumpyClient
        elif args.clientMode == "fedprox":
            target_class = prox.CustomNumpyClient
        elif args.clientMode == "fedopt":
            target_class = opt.CustomNumpyClient
        elif args.clientMode == "fedeve":
            target_class = eve.CustomNumpyClient
            # 영속성 모멘텀 변수 매핑 및 복원
            if "v_local" not in GLOBAL_CLIENT_HISTORIES[self.cid]:
                GLOBAL_CLIENT_HISTORIES[self.cid]["v_local"] = [
                    torch.zeros_like(p, device=self.DEVICE, requires_grad=False) for p in self.net.parameters()
                ]
            self.v_local = GLOBAL_CLIENT_HISTORIES[self.cid]["v_local"]
            
        elif args.clientMode == "adabest":
            target_class = adabest.CustomNumpyClient
            # 영속성 바이어스 변수 매핑 및 복원
            if "h_local" not in GLOBAL_CLIENT_HISTORIES[self.cid]:
                GLOBAL_CLIENT_HISTORIES[self.cid]["h_local"] = [
                    torch.zeros_like(p, device=self.DEVICE, requires_grad=False) for p in self.net.parameters()
                ]
            self.h_local = GLOBAL_CLIENT_HISTORIES[self.cid]["h_local"]
        else:
            raise ValueError(f"지원하지 않는 clientMode입니다: {args.clientMode}")

        # 2. [핵심] 런타임 인스턴스 중복 생성을 막고, 해당 클래스의 함수 본체를 
        # '현재 인스턴스(self)'에 완벽하게 바인딩 (types.MethodType 활용)
        self.fit = types.MethodType(target_class.fit, self)
        self.set_parameters = types.MethodType(target_class.set_parameters, self)
        self.get_parameters = types.MethodType(target_class.get_parameters, self)
        
        # 만약 evaluate가 타깃 클래스에 정의되어 있다면 바인딩, 없으면 기본값 유지
        if hasattr(target_class, "evaluate"):
            self.evaluate = types.MethodType(target_class.evaluate, self)

    def fit(self, parameters, config={}):
        # 이 메소드는 __init__ 단계에서 타깃 알고리즘의 fit으로 대체(오버라이딩)됩니다.
        pass