import torch.nn as nn
import torch.nn.functional as F
import torch
class tinyNet(nn.Module):
    def __init__(self, outdim=10):
        super(tinyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)# 3개의 입력 채널, 32개의 출력 채널, 3x3 커널 크기, 패딩=1
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()  # ReLU 활성화 함수
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 32개의 입력 채널, 64개의 출력 채널, 3x3 커널 크기, 패딩=1
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 커널 크기와 2의 스트라이드를 사용하는 Max Pooling
        self.flatten = nn.Flatten()  # 완전 연결 계층(FC Layer)을 통과하기 위한 텐서 평탄화
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 64x8x8의 입력 특성, 512개의 출력 특성을 가지는 완전 연결층

        # 64: 두 번째 합성곱 레이어에서 출력된 채널 수입니다. 즉, 각 위치에서 64개의 특성 맵이 생성되었습니다.
        # 8x8: 두 번째 합성곱 레이어의 출력 특성맵의 공간 크기입니다. 이는 입력 이미지의 공간 해상도가 합성곱과 풀링 연산을 거치면서 감소한 결과입니다.

        self.relu3 = nn.ReLU()  # ReLU 활성화 함수
        self.fc2 = nn.Linear(512, outdim)  # 512개의 입력 특성, 10개(레이블 개수) 출력 특성을 가지는 완전 연결층
        self.dropout = nn.Dropout(0.1)  # 드롭아웃 레이어, 50% 확률로 뉴런을 비활성화
    def forward(self, x):
        x = self.conv1(x)  # 입력: (N, 3, 32, 32), 출력: (N, 32, 32, 32) N은 배치 사이즈
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.pool(x)  # 입력: (N, 32, 32, 32), 출력: (N, 32, 16, 16)
        x = self.conv2(x)  # 입력: (N, 32, 16, 16), 출력: (N, 64, 16, 16)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.pool(x)  # 입력: (N, 64, 16, 16), 출력: (N, 64, 8, 8)
        x = self.flatten(x)  # 2D 텐서로 평탄화, 입력: (N, 64, 8, 8), 출력: (N, 64*8*8)
        x = self.fc1(x)  # 입력: (N, 64*8*8), 출력: (N, 512)
        x = self.relu3(x)
        x = self.fc2(x)  # 입력: (N, 512), 출력: (N, 10)
        return x
