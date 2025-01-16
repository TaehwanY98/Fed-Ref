from typing import Dict, List, Optional, Tuple
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr
import warnings
from flwr.common.logger import log
from flwr.server.strategy.fedavg import aggregate, aggregate_inplace
from torch.utils.data import DataLoader
from torch import save
from train import valid , make_model_folder, CustomFocalDiceLoss
import warnings
from utils.parser import Federatedparser
from utils.CustomDataset import Fets2022
import numpy as np
from Network.Unet import *
from Network.Loss import *
import getpass
import sys
user = getpass.getuser()
sys.path.append(f"/home/{user}/MICCAI/Network")
from client import seeding
from logging import WARNING
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors 
import pandas as pd
import torch
from functools import reduce

args = Federatedparser()
seeding(args)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aggregated_net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
aggregated_net.to(DEVICE)

ref_net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
ref_net.to(DEVICE)

dataset = Fets2022(args.client_dir)
# _, validset = random_split(dataset, [0.9, 0.1],  torch.Generator().manual_seed(args.seed))
validLoader = DataLoader(dataset, args.batch_size, False, collate_fn=lambda x: x)
lossf = CustomFocalDiceLoss()
OMEGA = 0.7

def ndarrays_to_arrays(param):
    return [v.flatten() for v in param]

def cosine_similarity_cal(X, Y):
    cosine = [cosine_similarity(x.reshape(1, -1),y.reshape(1, -1)) if len(x)>1 else x-y for x,y in zip(X,Y)]
    return [float(param[0]) for param in cosine]

def comparing_net(ref_net, aggregated_net):
    history1=valid(aggregated_net, validLoader, 0, lossf, DEVICE)
    history2=valid(ref_net, validLoader, 0, lossf, DEVICE)
    if history1["mDice"]>history2["mDice"]:
        return True
    else:
        return False
    
def cosine_distance_cal(X,Y):
    cosine = [cosine_distances(x.reshape(1, -1),y.reshape(1,-1))[0] if len(x)>1 else x-y for x,y in zip(X,Y)]
    return [float(param[0]) for param in cosine]

def Adaptive_eps(data, min_samples):
    # 1. Estimate local density
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(data)
    distances, _ = neighbors.kneighbors(data)
    distances = distances[:, -1]  # Distances to k-th nearest neighbor
    # 2. Calculate adaptive epsilon
    epsilon = distances  # Adjust this factor as needed
    return float(epsilon.mean())
    
class ProposedFedAvg(flwr.server.strategy.FedAvg):
    def __init__(self, *, fraction_fit = 1, fraction_evaluate = 1, min_fit_clients = 2, min_evaluate_clients = 2, min_available_clients = 2, evaluate_fn = None, on_fit_config_fn = None, on_evaluate_config_fn = None, accept_failures = True, initial_parameters = None, fit_metrics_aggregation_fn = None, evaluate_metrics_aggregation_fn = None, inplace = True):
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        
    def aggregate_fit(self, server_round, results, failures):
        clusters={}
        
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            ref_ndarrays = {k:v.cpu().detach().numpy() for k, v in ref_net.state_dict().items()}
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            ref_ndarrays = {k:v.cpu().detach().numpy() for k, v in ref_net.state_dict().items()}
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            total_sample = sum([fit_res.num_examples
                for _, fit_res in results])
            
            weighted_weights = [[layer * num_examples / total_sample for layer in weights] for weights, num_examples in weights_results]
            
            refp = ndarrays_to_arrays(ref_ndarrays.values())

            for indx, client_params in  enumerate([ndarrays_to_arrays(params) for params in weighted_weights]):
                clusters[f"client{indx+1}"] = cosine_distance_cal(refp, ndarrays_to_arrays(client_params))
            
            clusterv = np.array(list(clusters.values()))
            data = clusterv
            Dbscan = DBSCAN(eps=Adaptive_eps(data, 2), min_samples=2)
            cluster_index = Dbscan.fit_predict(data)
            
            print(cluster_index)
            
            uniq, count = np.unique(cluster_index, return_counts=True)
            
            ap = map(lambda v : ndarrays_to_arrays(v), weighted_weights)
            dictionary = {u : 0 for u in uniq} 
            
            for label, p in zip(cluster_index, ap):
                S = cosine_similarity_cal(p, refp)
                dictionary[label] += sum(S)/len(S)
                
            weighted_sil = {u : 0 for u in uniq}
            for u, c in zip(uniq, count):
                weighted_sil[int(u)] = dictionary[int(u)]/int(c)
            print(list(weighted_sil.values())) 
            
            selected_uniq = [key for key, value in weighted_sil.items() if value > args.alpha]
            selected_index = reduce(lambda x,y : np.bitwise_or(x,y), [cluster_index==key for key in selected_uniq], [False]*len(cluster_index))
            
            if server_round in [0, 1, 2]:
                aggregated_ndarrays = aggregate_inplace(results)
            else:
                aggregated_ndarrays = aggregate([client_sample for client_sample, bool in zip(weights_results, selected_index) if bool])
       
        set_parameters(aggregated_net, aggregated_ndarrays)
        if server_round in [0,1,2]:
            bool = True
        else:
            bool = comparing_net(ref_net, aggregated_net)
        
        if bool:
        
            DV = [rp-ap for ap, rp in zip(ndarrays_to_arrays(aggregated_ndarrays), ndarrays_to_arrays(ref_ndarrays.values()))]
        
            similar = cosine_similarity_cal(ndarrays_to_arrays(aggregated_ndarrays), ndarrays_to_arrays(ref_ndarrays.values()))
        
            proposed_eq = [rp-(dv*(1-np.abs(sim))*OMEGA) for rp, dv, sim in zip(ndarrays_to_arrays(ref_ndarrays.values()), DV, similar)]
            
            set_parameters(ref_net, proposed_eq)
            
            parameters_eq = ndarrays_to_parameters([w.cpu().detach().numpy()  for w in aggregated_net.parameters()])
            
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            return parameters_eq, metrics_aggregated
        else:
        
            DV = [ap-rp for ap, rp in zip(ndarrays_to_arrays(aggregated_ndarrays), ndarrays_to_arrays(ref_ndarrays.values()))]
        
            similar = cosine_similarity_cal(zip(ndarrays_to_arrays(aggregated_ndarrays), ndarrays_to_arrays(ref_ndarrays.values())))
        
            
            proposed_eq = [ap-(dv*(1-np.abs(sim))*OMEGA) for ap, dv, sim in zip(ndarrays_to_arrays(aggregated_ndarrays), DV, similar)]
            
            set_parameters(aggregated_net, proposed_eq)
            
            parameters_eq = ndarrays_to_parameters([w.cpu().detach().numpy() for w in aggregated_net.parameters()])

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            return parameters_eq, metrics_aggregated

def set_parameters(net, new_parameters):
    for old, new in zip(net.parameters(), new_parameters):
        shape = old.data.size()
        old.data = torch.Tensor(new).view(shape).to(DEVICE)
    
def fl_save(server_round:int, parameters: flwr.common.NDArrays, config:Dict[str, flwr.common.Scalar], validF=valid)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
    set_parameters(aggregated_net, parameters)
    save(aggregated_net.state_dict(), f"./Models/{args.version}/net.pt")
    print("model is saved")
    return 0, {}

def fl_evaluate(server_round:int, parameters: flwr.common.NDArrays, config:Dict[str, flwr.common.Scalar], validF=valid)-> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:
    set_parameters(aggregated_net, parameters)
    history=validF(aggregated_net, validLoader, 0, CustomFocalDiceLoss(), DEVICE)
    save(aggregated_net.state_dict(), f"./Models/{args.version}/net.pt")
    return history['loss'], {key:value for key, value in history.items() if key != "loss" }

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    make_model_folder(f"./Models/{args.version}")
    
    
    history = flwr.server.start_server(
        server_address='[::]:8084',strategy=ProposedFedAvg(evaluate_fn=fl_evaluate, inplace=False, min_fit_clients=7, min_available_clients=7, min_evaluate_clients=7), 
                           config=flwr.server.ServerConfig(num_rounds=args.round)
    )
    pd.DataFrame(history.metrics_centralized).to_csv("./Result/FedRef.csv", index=False)
    pd.DataFrame(history.losses_centralized).to_csv("./Result/FedRef_loss.csv", index=False)