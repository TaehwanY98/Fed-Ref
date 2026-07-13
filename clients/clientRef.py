import flwr 
import types  # 동적 메서드 바인딩을 위한 빌트인 모듈
import torch
# 사용자 정의 모듈 임포트
import client as avg
import clientProxy as prox
import clientOpt as opt
import clientFedEve as eve
import clientAdaBest as adabest

class CustomNumpyClient(flwr.client.NumPyClient):
    def __init__(self, net, train_loader, length, epoch, lossf, optimizer, DEVICE, args, trainF=lambda x: x, validF=lambda x: x):
        super().__init__()
        
        # 공통 기본 속성 정의
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

        # 1. args.clientMode 조건에 따라 매핑할 소스 클래스(또는 모듈) 결정
        target_source = None
        
        if args.clientMode == "fedavg":
            target_source = avg.CustomNumpyClient
        elif args.clientMode == "fedprox":
            target_source = prox.CustomNumpyClient
        elif args.clientMode == "fedopt":
            target_source = opt.CustomNumpyClient
        elif args.clientMode == "fedeve":
            target_source = eve.FedEveClient
            # FedEve 전용 내부 속성 초기화
            self.v_local = [torch.zeros_like(p, device=self.DEVICE) for p in self.net.parameters()]
        elif args.clientMode == "adabest":
            target_source = adabest.AdaBestClient
            # AdaBest 전용 내부 속성 초기화
            self.h_local = [torch.zeros_like(p, device=self.DEVICE) for p in self.net.parameters()]
        else:
            raise ValueError(f"지원하지 않는 clientMode입니다: {args.clientMode}")

        # 2. 결정된 대상 클래스로부터 핵심 메서드들을 현재 인스턴스(self)에 바인딩
        # 이 작업을 거치면 각 파일에 적힌 로직(예: proxy_lossf, adabest_lossf 등)이 self를 통해 작동합니다.
        methods_to_bind = ["fit", "evaluate", "set_parameters", "get_parameters"]
        
        for method_name in methods_to_bind:
            if hasattr(target_source, method_name):
                # 중요: 클래스의 함수를 인스턴스(self)의 메서드로 묶어줍니다 (Method Binding)
                func = getattr(target_source, method_name)
                setattr(self, method_name, types.MethodType(func, self))