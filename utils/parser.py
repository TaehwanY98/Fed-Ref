import argparse

def Simulationparser():
    parser= argparse.ArgumentParser(
        prog="Federated Learning in fets2022",
        description="Federated Learning code by using fets2022 Dataset",
    )
    # Required
    parser.add_argument("-r", "--round", type= int, default=10)
    parser.add_argument("-e", "--epoch", type= int, default= 3)
    parser.add_argument("-bs", "--batch-size", type= int, default=256)
    parser.add_argument("-l", "--lr", type= float, default= 1e-5)
    parser.add_argument("-udp", "--degrade", type= float, default= 0.3)
    parser.add_argument("-t", "--type", type= str, default="femnist")
    parser.add_argument("-m", "--mode", type= str, default="fedavg")
    parser.add_argument('-NON',"--numberOfNodes", type= int, default= 10)
    parser.add_argument("-g", "--gpu", type= bool, default= True)
    parser.add_argument("-rp", "--result-path", type= str, default="Result")
    parser.add_argument("--token", type=str, default= "")
    parser.add_argument("-v", "--version", type= str, default="default")
    parser.add_argument("-s", "--seed", type= int, default= 2024)
    args = parser.parse_args()
    # Optional
    if args.type in ["fets"]:
        parser.add_argument("-d", "--data-dir", type= str, default=None)
        parser.add_argument("-cd", "--client-dir", type= str, default=None)
        args.batch_size = 1
    if args.mode == "fedref":
        parser.add_argument("--lda", type= float, default= 0.1)
        parser.add_argument("--delta", type= float, default= 0.0)
        parser.add_argument("-p", "--prime", type= int, default=2)
        parser.add_argument("-cm", "--clientMode",type= str, default="fedavg")
    elif args.mode == "fedprox":
        parser.add_argument("--mu", type= float, default= 0.5)
    elif args.mode == "fedopt":
        parser.add_argument("--beta1", type= float, default= 0.9)
        parser.add_argument("--beta2", type= float, default= 0.99)
        parser.add_argument("--tau", type= float, default= 1e-4)
    elif args.mode == "adabest":
        pass
    elif args.mode == "fedeve":
        pass
    parser.add_argument("--test", type= bool, default= False)
    # parser.add_argument("-i", "--id", type= int, default=1)
    args = parser.parse_args()
    return args