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
    
class FedRef(flwr.server.strategy.FedAvg):
    def __init__(self, ref_net:nn.Module, aggregated_net:nn.Module,dataset, validLoader, args, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.ref_net = ref_net
        self.aggregated_net = aggregated_net
        self.dataset = dataset
        self.validLoader = validLoader
        self.args = args
        self.evaluate_fn = self.evaluate_fn
    def aggregate_fit(self, server_round, results, failures):
        clusters={}
        
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            ref_ndarrays = [param.cpu().detach().numpy() for param in self.ref_net.parameters()]
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            ref_ndarrays = [param.cpu().detach().numpy() for param in self.ref_net.parameters()]
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            total_sample = sum([int(fit_res.num_examples)
                for _, fit_res in results])
            
            weighted_weights = [[layer * int(num_examples) / total_sample for layer in weights] for weights, num_examples in weights_results]
            
            for indx, client_params in  enumerate(weighted_weights):
                clusters[f"client{indx+1}"] = cosine_distance_cal(ref_ndarrays, client_params)
            
            clusterv = np.array(list(clusters.values()))
            data = clusterv
            if self.args.type != "drive":
                Dbscan = DBSCAN(eps=Adaptive_eps(ndarrays_to_arrays(data), 2), min_samples=2)
            else:
                Dbscan = DBSCAN(eps=0.1, min_samples=2)
            cluster_index = Dbscan.fit_predict(ndarrays_to_arrays(data))
            
            print(cluster_index)
            
            uniq, count = np.unique(cluster_index, return_counts=True)
            
            ap = map(lambda v : ndarrays_to_arrays(v), weighted_weights)
            dictionary = {u : 0 for u in uniq} 
            
            for label, p in zip(cluster_index, ap):
                S = cosine_similarity_cal(p, ref_ndarrays)
                dictionary[label] += sum(S)/len(S)
                
            weighted_sil = {u : 0 for u in uniq}
            for u, c in zip(uniq, count):
                weighted_sil[int(u)] = dictionary[int(u)]/int(c)
            print(list(weighted_sil.values())) 
            
            selected_uniq = [key for key, value in weighted_sil.items() if value > self.args.alpha]
            selected_index = reduce(lambda x,y : np.bitwise_or(x,y), [cluster_index==key for key in selected_uniq], [False]*len(cluster_index))
            
            if server_round in [0, 1, 2]:
                aggregated_ndarrays = aggregate_inplace(results)
            else:
                aggregated_ndarrays = aggregate([client_sample for client_sample, bool in zip(weights_results, selected_index) if bool])
       
        set_parameters(self.aggregated_net, aggregated_ndarrays)
        if server_round in [0,1,2]:
            bool = True
        else:
            if self.args.type == "octdl":
                bool = comparing_net(self.ref_net, self.aggregated_net, "loss", self.dataset, self.validLoader, self.args)
            else:
                bool = comparing_net(self.ref_net, self.aggregated_net, metric, self.dataset, self.validLoader, self.args)
        
        if bool:
        
            DV = [rp-ap for ap, rp in zip(ndarrays_to_arrays(aggregated_ndarrays), ndarrays_to_arrays(ref_ndarrays))]
        
            similar = cosine_similarity_cal(ndarrays_to_arrays(aggregated_ndarrays), ndarrays_to_arrays(ref_ndarrays))
        
            proposed_eq = [rp-(dv*(1-np.abs(sim))*OMEGA) for rp, dv, sim in zip(ndarrays_to_arrays(ref_ndarrays), DV, similar)]
            
            set_parameters(self.ref_net, proposed_eq)
            
            parameters_eq = ndarrays_to_parameters([w.cpu().detach().numpy()  for w in self.aggregated_net.parameters()])
            
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            return parameters_eq, metrics_aggregated
        else:
        
            DV = [ap-rp for ap, rp in zip(ndarrays_to_arrays(aggregated_ndarrays), ndarrays_to_arrays(ref_ndarrays))]
        
            similar = cosine_similarity_cal(ndarrays_to_arrays(aggregated_ndarrays), ndarrays_to_arrays(ref_ndarrays))
        
            
            proposed_eq = [ap-(dv*(1-np.abs(sim))*OMEGA) for ap, dv, sim in zip(ndarrays_to_arrays(aggregated_ndarrays), DV, similar)]
            
            set_parameters(self.aggregated_net, proposed_eq)
            
            parameters_eq = ndarrays_to_parameters([w.cpu().detach().numpy() for w in self.aggregated_net.parameters()])

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            return parameters_eq, metrics_aggregated
    def evaluate(self, server_round: int, parameters)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
        parameters = parameters_to_ndarrays(parameters)
        lossf = CustomFocalDiceLoss() if not self.args.type=="octdl" else nn.BCEWithLogitsLoss(self.dataset.label_weight)
        if self.args.type in ["drive"]:
            lossf = CustomFocalDiceLossb()
        validF= valid if not self.args.type=="octdl" else octValid
        if self.args.type in ["drive"]:
            validF = validDrive
        set_parameters(self.aggregated_net, parameters)
        history=validF(self.aggregated_net, self.validLoader, 0, lossf.to(DEVICE), DEVICE, True)
        make_dir(self.args.result_path)
        make_dir(os.path.join(self.args.result_path, self.args.mode))
        if server_round != 0:
            old_historyframe = pd.read_csv(os.path.join(self.args.result_path, self.args.mode, f'FedRef_{self.args.type}.csv'))
            historyframe = pd.DataFrame({k:[v] for k, v in history.items()})
            newframe=pd.concat([old_historyframe, historyframe])
            newframe.to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedRef_{self.args.type}.csv'), index=False)
        else:
            pd.DataFrame({k:[v] for k, v in history.items()}).to_csv(os.path.join(self.args.result_path, self.args.mode, f'FedAvg_{self.args.type}.csv'), index=False)
        save(self.aggregated_net.state_dict(), f"./Models/{self.args.version}/net.pt")
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