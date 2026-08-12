import argparse

def Simulationparser():
    # 1. 메인 파서 정의 (기본 인자들)
    parser = argparse.ArgumentParser(
        prog="Federated Learning",
        description="Federated Learning code",
    )
    
    # Required / 기본 인자 등록
    parser.add_argument("-r", "--round", type=int, default=10)
    parser.add_argument("-e", "--epoch", type=int, default=3)
    parser.add_argument("-bs", "--batch-size", type=int, default=256)
    parser.add_argument("-l", "--lr", type=float, default=1e-5)
    parser.add_argument("-udp", "--degrade", type=float, default=0.3)
    parser.add_argument("-t", "--type", type=str, default="femnist")
    parser.add_argument("-m", "--mode", type=str, default="fedavg")
    parser.add_argument('-NON', "--numberOfNodes", type=int, default=10)
    
    # ⚠️ 중요: bool 타입은 action="store_true"를 사용하는 것이 안전합니다.
    parser.add_argument("-g", "--gpu", action="store_true",help="Use GPU if flagged (default: False)")
    parser.add_argument("-rp", "--result-path", type=str, default="Result")
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("-v", "--version", type=str, default="default")
    parser.add_argument("-s", "--seed", type=int, default=2024)
    parser.add_argument("--test", action="store_true", help="Run in test mode")

    # 2. '모드(mode)'나 '타입(type)'에 의존적인 선택적 인자들을 미리 정의
    # (argparse는 파싱 중에 동적으로 인자를 추가하는 것을 지원하지 않습니다)
    parser.add_argument("-d", "--data-dir", type=str, default=None)
    parser.add_argument("-cd", "--client-dir", type=str, default=None)
    
    parser.add_argument("--lda", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("-p", "--prime", type=int, default=4)
    parser.add_argument("-cm", "--clientMode", type=str, default="fedavg")

    # 3. 단 한 번만 파싱을 수행합니다.
    args = parser.parse_args()
    
    # 4. 파싱된 이후의 값 조건부 변경 (예: fets일 때 batch_size 강제 변경)
    if args.type == "fets":
        args.batch_size = 1
        
    return args

# 테스트 실행
if __name__ == "__main__":
    args = Simulationparser()
    print(args)