from typing import Dict, List, Optional, Tuple
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr
import warnings
import numpy as np
from flwr.common.logger import log
from flwr.server.strategy.fedavg import aggregate, aggregate_inplace
from torch.utils.data import DataLoader
from torch import save
from utils.train import valid , make_model_folder, CustomFocalDiceLoss, CustomFocalDiceLossb, set_seeds, validDrive
from utils.octTrain import valid as octValid
from utils.MNISTTrain import valid as MNISTValid
from utils.CIFAR10Train import valid as CIFAR10valid
from utils.parser import Federatedparser
from utils.CustomDataset import Fets2022, BRATS, OCTDL
from Network.Unet import *
from Network.Loss import *
from Network.Resnet import ResNet
from logging import WARNING
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors 
import pandas as pd
import torch
from functools import reduce
import os
metric = "loss"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OMEGA = 0.7

def ndarrays_to_arrays(param):
    return [np.nan_to_num(v.flatten(), posinf=0.0, neginf=0.0, nan=0.0) for v in param]

def cosine_similarity_cal(X, Y):
    try:
        cosine = [cosine_similarity(x.reshape(1, -1),y.reshape(1, -1))[0]  for x,y in zip(X,Y)]
    except:
        cosine = [cosine_similarity(x,y)[0]  for x,y in zip(X,Y)]
    return [float(param[0]) for param in cosine]

def comparing_net(ref_net, aggregated_net, metric, dataset, validLoader,args):
    validF=valid if not args.type=="octdl" else octValid
    if args.type == "drive":
        validF = validDrive
    lossf = CustomFocalDiceLoss() if not args.type=="octdl" else nn.BCEWithLogitsLoss(dataset.label_weight)
    if args.type == "drive":
        lossf = CustomFocalDiceLossb()
    history1=validF(aggregated_net, validLoader, 0, lossf.to(DEVICE), DEVICE)
    history2=validF(ref_net, validLoader, 0, lossf.to(DEVICE), DEVICE)

    if history1[metric]>history2[metric]:
        if 'loss' in metric:
            return False
        return True
    else:
        if 'loss' in metric:
            return True
        return False
    
def cosine_distance_cal(X,Y):
    try:
        cosine = [cosine_distances(x.reshape(1, -1),y.reshape(1,-1))[0] for x,y in zip(X,Y)]
    except:
        cosine = [cosine_distances(x,y)[0] for x,y in zip(X,Y)]
    return [param.astype("float32") for param in cosine]

def Adaptive_eps(data, min_samples):
    # 1. Estimate local density
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(data)
    distances, _ = neighbors.kneighbors(data)
    distances = distances[:, -1]  # Distances to k-th nearest neighbor
    # 2. Calculate adaptive epsilon
    epsilon = distances  # Adjust this factor as needed
    return float(np.nan_to_num(epsilon.mean(), nan=0.0, posinf=0.1, neginf=0.1))

class CircleQueue():
    def __init__(self, max_queue_size):
        self.front = 0
        self.rear = 0
        self.items = [None]* max_queue_size
        self.max_que_size = max_queue_size
    
    def isEmpty(self):
        return self.front == self.rear
    def isFull(self):
        return self.front == (self.rear+1)%self.max_que_size    
    def clear(self):
        self.front = self.rear
    def enqueue(self, item):
        if not self.isFull():
            self.rear = (self.rear + 1) % self.max_que_size
            self.items[self.rear] = item
        else:
            self.dequeue()
            self.rear = (self.rear + 1) % self.max_que_size
            self.items[self.rear] = item
    def dequeue(self):
        if not self.isEmpty():
            self.front = (self.front+1) % self.max_que_size
            return self.items[self.front]
        else:
            pass

class FedRef(flwr.server.strategy.FedAvg):
    def __init__(self, ref_net:nn.Module, aggregated_net:nn.Module, lossf, dataset, validLoader, args, p:int=2, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.ref_net = ref_net
        self.theta0 ={"agg":[], "ref":[]}
        self.aggregated_net = aggregated_net
        self.lossf = lossf
        self.dataset = dataset
        self.validLoader = validLoader
        self.args = args
        self.evaluate_fn = self.evaluate_fn
        self.p = p
        self.aggs = CircleQueue(p)
        self.losses = CircleQueue(1)
        
    def SetTheta0(self, agg_ndarrays, ref_ndarrays):
        self.theta0["agg"] = agg_ndarrays
        self.theta0["ref"] = ref_ndarrays
        
        
    def aggregate_fit(self, server_round, results, failures):
        
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            if server_round < self.p+1:
                # Convert results
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results
                ]
                aggregated_ndarrays = aggregate(weights_results)
                self.aggs.enqueue(aggregated_ndarrays)
                self.SetTheta0(aggregated_ndarrays, [])
                aggLosses = [res.metrics["loss"] for _,res in results]
                self.losses.enqueue(aggLosses)
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
                # Aggregate custom metrics if aggregation fn was provided
                metrics_aggregated = {}
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                elif server_round == 1:  # Only log this warning once
                    log(WARNING, "No fit_metrics_aggregation_fn provided")
                
            else:
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results
                ]
                aggregated_ndarrays = aggregate(weights_results)
                self.aggs.enqueue(aggregated_ndarrays)
                
                aggLosses = [res.metrics["loss"] for _,res in results]
                aggExampls = [res.num_examples for _,res in results]
                aggTotalExamples = sum(aggExampls)
                aggWeights = np.array(aggExampls)/aggTotalExamples
                ref_ndarrays = [[layer for layer in weights] for weights in self.aggs.items]
                ref_ndarrays_sq = [(reduce(np.add, layer_updates) / self.aggs.max_que_size)-t0 for layer_updates, t0 in zip(zip(*ref_ndarrays), self.theta0["ref"])]
                agg_ndarrays_sq = [layer_updates-t0 for layer_updates, t0 in zip(aggregated_ndarrays, self.theta0["agg"])]
                
                
                self.SetTheta0(aggregated_ndarrays, ref_ndarrays)
                metrics_aggregated = {}
                
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                parameters_aggregated = self.BayesianTransferLearning(aggregated_ndarrays, self.args.lr, p1Losses=aggLosses, preLosses=self.losses.items[0], p1Weights=aggWeights,target1_netL1=agg_ndarrays_sq ,target2_netL1=ref_ndarrays_sq, Lambda=self.args.lda)
                parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)
                self.losses.enqueue(aggLosses)
                
        return parameters_aggregated, metrics_aggregated
        
    def BayesianTransferLearning(self, p1, lr, p1Losses, preLosses, p1Weights, target1_netL1, target2_netL1, Lambda=0.2):
        p1 = [W1 - lr*(reduce(np.add, (p1Weights)/len(p1Losses)*((reduce(np.add, p1Losses)) - reduce(np.add, preLosses))) +Lambda*np.abs(W3) + Lambda*np.abs(W2)) for W1, W2, W3 in zip(p1, target2_netL1, target1_netL1)]
        return p1 
        
    def evaluate(self, server_round: int, parameters)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
        parameters = parameters_to_ndarrays(parameters)
        validF= valid if not self.args.type=="octdl" else octValid
        if self.args.type == "mnist":
            validF = MNISTValid
        if self.args.type in ["drive"]:
            validF = validDrive
        if self.args.type == "cifar10":
            validF = CIFAR10valid
        set_parameters(self.aggregated_net, parameters)
        history=validF(self.aggregated_net, self.validLoader, 0, self.lossf.to(DEVICE), DEVICE, True)
        make_dir(self.args.result_path)
        make_dir(os.path.join(self.args.result_path, self.args.mode))
        if server_round != 0:
            old_historyframe = pd.read_csv(os.path.join(self.args.result_path, self.args.mode, f'FedRef_{self.args.type}_lda{self.args.lda*10}_p{self.args.prime}.csv'))
            historyframe = pd.DataFrame({k:[v] for k, v in history.items()})
            newframe=pd.concat([old_historyframe, historyframe])
            newframe.to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedRef_{self.args.type}_lda{self.args.lda*10}_p{self.args.prime}.csv'), index=False)
        else:
            pd.DataFrame({k:[v] for k, v in history.items()}).to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedRef_{self.args.type}_lda{self.args.lda*10}_p{self.args.prime}.csv'), index=False)
        save(self.aggregated_net.state_dict(), f"./Models/{self.args.version}/net_lda{self.args.lda*10}_p{self.args.prime}.pt")
        return history['loss'], {key:value for key, value in history.items() if key != "loss" }
        
        # return super().aggregate_fit(server_round, results, failures)
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    
def set_parameters(net, new_parameters):
    for old, new in zip(net.parameters(), new_parameters):
        shape = old.data.size()
        old.data = torch.Tensor(new).view(shape).to(DEVICE)

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    args = Federatedparser()
    set_seeds(args)
    make_model_folder(f"./Models/{args.version}")
    
    if args.type in ["fets", "brats"]:
        aggregated_net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
        aggregated_net.to(DEVICE)
        ref_net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
        ref_net.to(DEVICE)
    elif args.type in "octdl":
        aggregated_net = ResNet()
        aggregated_net.to(DEVICE)
        ref_net = ResNet()
        ref_net.to(DEVICE)
    if args.type == "fets":
        dataset = Fets2022(args.client_dir)
    if args.type == "brats":
        dataset = BRATS(args.client_dir)
    if args.type == "octdl":
        dataset = OCTDL(args.client_dir)
    validLoader = DataLoader(dataset, args.batch_size, False, collate_fn=lambda x: x)
    
    history = flwr.server.start_server(
        server_address='[::]:8084',strategy=FedRef(ref_net, aggregated_net, dataset, validLoader, args, inplace=False, min_fit_clients=7, min_available_clients=7, min_evaluate_clients=7), 
                           config=flwr.server.ServerConfig(num_rounds=args.round)
    )
    if args.type == 'fets':
        pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedRef_fets.csv", index=False)
        pd.DataFrame(history.losses_centralized).to_csv("./Result/FedRef_loss_fets.csv", index=False)
    elif args.type == "brats":
        pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedRef_BRATS.csv", index=False)
        pd.DataFrame(history.losses_centralized).to_csv("./Result/FedRef_loss_BRATS.csv", index=False)
    elif args.type == "octdl":
        pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedRef_OCTDL.csv", index=False)
        pd.DataFrame(history.losses_centralized).to_csv("./Result/FedRef_loss_OCTDL.csv", index=False)