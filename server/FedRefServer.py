from typing import Dict, List, Optional, Tuple
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr
from torch import nn
import numpy as np
from flwr.common.logger import log
from flwr.server.strategy.fedavg import aggregate, aggregate_inplace
from torch import save
from utils.TumorTrain import valid , validDrive
from utils.octTrain import valid as octValid
from utils.MNISTTrain import valid as MNISTValid
from utils.CIFAR10Train import valid as CIFAR10valid
from logging import WARNING
import pandas as pd
import torch
from functools import reduce
import os
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.rear = (self.rear + 1) % self.max_que_size
        self.items[self.rear] = item
        

class FedRef(flwr.server.strategy.FedAvg):
    def __init__(self, ref_net:nn.Module, aggregated_net:nn.Module, lossf, validLoader, args, p:int=2, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.ref_net = ref_net
        self.theta0 ={"agg":[], "ref": [val.cpu().detach().numpy() for val in self.ref_net.parameters()]}
        self.aggregated_net = aggregated_net
        self.lossf = lossf
        self.validLoader = validLoader
        self.args = args
        self.evaluate_fn = self.evaluate_fn
        self.p = p
        self.aggs = CircleQueue(p)
        self.losses = [None]
        
    def SetTheta0(self, agg_ndarrays, ref_ndarrays):
        self.theta0["agg"] = agg_ndarrays
        if ref_ndarrays is not None:
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
                self.aggs.enqueue(copy.deepcopy(aggregated_ndarrays))
                self.SetTheta0(copy.deepcopy(aggregated_ndarrays), None)
                aggLosses = [res.metrics["loss"] for _,res in results]
                self.losses[0]=aggLosses
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
                # Aggregate custom metrics if aggregation fn was provided
                metrics_aggregated = {}
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                elif server_round == 1:  # Only log this warning once
                    log(WARNING, "No fit_metrics_aggregation_fn provided")
                return parameters_aggregated, metrics_aggregated
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
                
                if server_round>self.p+1:
                    ref_ndarrays_sq = [(reduce(np.add, layer_updates) / self.aggs.max_que_size)-(reduce(np.add, t0) / self.aggs.max_que_size) for layer_updates, t0 in zip(zip(*ref_ndarrays), self.theta0["ref"])]
                    agg_ndarrays_sq = [layer_updates-t0 for layer_updates, t0 in zip(aggregated_ndarrays, self.theta0["agg"])]
                
                metrics_aggregated = {}
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                    
                
                if server_round>self.p+1:
                    out = self.BayesianTransferLearning(aggregated_ndarrays, self.args.lr, p1Losses=aggLosses, preLosses=self.losses[0], p1Weights=aggWeights,target1_netL1=agg_ndarrays_sq ,target2_netL1=ref_ndarrays_sq, Lambda=self.args.lda)
                    parameters_aggregated = ndarrays_to_parameters(out)
                else:
                    parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
                self.losses[0]=aggLosses
                self.SetTheta0([layer for layer in aggregated_ndarrays], [reduce(np.add, layer) / self.aggs.max_que_size for layer in zip(*ref_ndarrays)])
                return parameters_aggregated, metrics_aggregated
        
    def BayesianTransferLearning(self, p1, lr, p1Losses, preLosses, p1Weights, target1_netL1, target2_netL1, Lambda=0.2):
        p1 = [W1 - lr*(reduce(np.add, (p1Weights)/len(p1Losses)*((reduce(np.add, p1Losses)) - reduce(np.add, preLosses))) +Lambda*np.linalg.norm(W3.flatten(),2) + Lambda*np.linalg.norm(W2.flatten(), 2)) for W1, W2, W3 in zip(p1, target2_netL1, target1_netL1)]
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
        
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    
def set_parameters(net, new_parameters):
    for old, new in zip(net.parameters(), new_parameters):
        shape = old.data.size()
        old.data = torch.Tensor(new).view(shape).to(DEVICE)