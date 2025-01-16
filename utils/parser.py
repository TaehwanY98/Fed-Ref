import argparse

def Centralparser():
    parser= argparse.ArgumentParser(
        prog="Central Learning in WESAD",
        description="centralized training code by using WESAD Dataset",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("-s", "--seed", type= int, default= 2024)
    parser.add_argument("-e", "--epoch", type= int, default= 10)
    parser.add_argument("-bs", "--batch-size", type= int, default=1)
    parser.add_argument("-l", "--lr", type= float, default= 1e-2)
    parser.add_argument("-d", "--data-dir", type= str, default=None, required=True)
    parser.add_argument("-p", "--pretrained", type= bool, default=False)
    args = parser.parse_args()
    return args

def Federatedparser():
    parser= argparse.ArgumentParser(
        prog="Federated Learning in fets2022",
        description="Federated Learning code by using fets2022 Dataset",
    )
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("--IPv4", type= str, required = True)
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