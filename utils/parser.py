import argparse

def Simulationparser():
    parser= argparse.ArgumentParser(
        prog="Federated Learning in fets2022",
        description="Federated Learning code by using fets2022 Dataset",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("--IPv4", type= str)
    parser.add_argument("-s", "--seed", type= int, default= 2024)
    parser.add_argument("-r", "--round", type= int, default=10)
    parser.add_argument("-e", "--epoch", type= int, default= 2)
    parser.add_argument("-i", "--id", type= int, default=1)
    parser.add_argument("-bs", "--batch-size", type= int, default=1)
    parser.add_argument("-d", "--data-dir", type= str, default=None, required=True)
    parser.add_argument("-cd", "--client-dir", type= str, default=None, required=True)
    parser.add_argument("-l", "--lr", type= float, default= 1e-5)
    parser.add_argument("--lda1", type= float, default= 0.1)
    parser.add_argument("--lda2", type= float, default= 0.1)
    parser.add_argument("--toplda2", type= float, default= 0.005)
    parser.add_argument("--sigmaw", type= float, default= 10.0)
    parser.add_argument("--sigmar", type= float, default= 10.0)
    parser.add_argument("-p", "--prime", type= int, default=2)
    parser.add_argument("-udp", "--degrade", type= float, default= 0.3)
    parser.add_argument("-t", "--type", type= str, default="fets")
    parser.add_argument("-m", "--mode", type= str, default="fedavg")
    parser.add_argument("-cn", "--client-num", type= int, default= 10)
    parser.add_argument("-g", "--gpu", type= bool, default= True)
    parser.add_argument("--test", type= bool, default= False)
    parser.add_argument("-rp", "--result-path", type= str, default="Result")
    parser.add_argument("--token", type=str, default= "")
    args = parser.parse_args()
    return args