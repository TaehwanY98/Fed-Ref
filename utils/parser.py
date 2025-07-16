import argparse

def Centralparser():
    parser= argparse.ArgumentParser(
        prog="Centralized Learning in fets2022",
        description="centralized training code by using fets2022 | Brats",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("-s", "--seed", type= int, default= 2024)
    parser.add_argument("-e", "--epoch", type= int, default= 10)
    parser.add_argument("-bs", "--batch-size", type= int, default=1)
    parser.add_argument("-l", "--lr", type= float, default= 1e-2)
    parser.add_argument("-d", "--data-dir", type= str, default=None, required=True)
    parser.add_argument("-p", "--pretrained", type= bool, default=False)
    parser.add_argument("-t", "--type", type= str, default="fets")
    args = parser.parse_args()
    return args

def Federatedparser():
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
    parser.add_argument("-l", "--lr", type= float, default= 1e-2)
    parser.add_argument("-a", "--alpha", type= float, default= 0.3)
    parser.add_argument("-p", "--pretrained", type= str, default=None)
    parser.add_argument("-t", "--type", type= str, default="fets")
    parser.add_argument("-m", "--mode", type= str, default="fedavg")
    parser.add_argument("-cn", "--client-num", type= int, default= 10)
    parser.add_argument("-g", "--gpu", type= bool, default= True)
    parser.add_argument("-rp", "--result-path", type= str, default="Result")
    args = parser.parse_args()
    return args

def Evaluateparaser():
    parser= argparse.ArgumentParser(
        prog="Evaluate model in fets2022",
        description="evaluate code by using fets2022 Dataset",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("-d", "--data-dir", type= str, default=None, required=True)
    parser.add_argument("-bs", "--batch-size", type= int, default=4)
    args = parser.parse_args()
    return args

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
    parser.add_argument("-l", "--lr", type= float, default= 1e-2)
    parser.add_argument("--lda", type= float, default= 0.1)
    parser.add_argument("-p", "--prime", type= int, default=2)
    parser.add_argument("-t", "--type", type= str, default="fets")
    parser.add_argument("-m", "--mode", type= str, default="fedavg")
    parser.add_argument("-cn", "--client-num", type= int, default= 10)
    parser.add_argument("-g", "--gpu", type= bool, default= True)
    parser.add_argument("--test", type= bool, default= False)
    parser.add_argument("-rp", "--result-path", type= str, default="Result")
    parser.add_argument("--token", type=str, default= "")
    args = parser.parse_args()
    return args