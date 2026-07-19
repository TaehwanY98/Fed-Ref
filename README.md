## Fed-Ref: Effective Unbounded Drift Control in Heterogeneous Federated Learning: Bayesian Fine-Tuning Using a Reference Model

### Abstract
Federated learning (FL) enables collaborative model training across distributed clients while preserving data privacy. However, data and system heterogeneity often induce unbounded drift in model updates due to multiple local steps, partial participation, and unbounded dynamics. These challenges severely degrade predictive performance, exacerbate catastrophic forgetting, and hinder global optimization. To address these limitations, we propose FedRef, a robust optimization framework that leverages a reference model generalized from prior global models via Bayesian fine-tuning. FedRef integrates a Maximum A Posteriori (MAP)-based regularization mechanism that calibrates global updates toward a temporally aggregated reference model, thereby effectively controlling unbounded drift and maximizing empirical stability.
Unlike prior approaches that rely on complex client-side optimization heuristics, FedRef achieves superior generalization by seamlessly blending a robust probabilistic prior with global likelihood optimization. By maintaining a temporal moving average centered on previous global trajectories, the proposed framework provides a highly reliable optimization anchor that effectively counteracts misleading updates under extreme unbounded drift settings. Extensive experiments on image classification (FEMNIST, CINIC-10) and medical image segmentation (FeTS2022) demonstrate that FedRef yields superior predictive performance, higher F1-scores, and significantly faster convergence under severely non-IID conditions. These results highlight FedRef as an exceptionally efficient and stable framework capable of unlocking high-fidelity generalization in heterogeneous real-world FL scenarios.

### Overview of our proposed FedRef system 
<img src="./res/FedRef.png" alt="overview-fedref" width="auto"/>


### Settings
| Environment set  | Settings for detail                             |
|------------------|-------------------------------------------------|
| FL framework     | Flower: a friendly federated learning framework |
| Language         | Python: 3.9.21                                  |
| Operation System | Linux 24.04 LTS                                 |
| GPU              | Nvidia RTX 4090                                 |
| Tools            | Visual studio code                              |


### Result

Will be updated..

### Run

Available Dataset
1. cinic10
2. femnist
3. fets * need custom data settings following: 


        ├── Folder (parameter: -cd ./Folder)
            ├── client1
            ├── client2
            ├── client3
            ├── client4
            ├── client5
            ├── client6
            ├── client7
            ├── client8
            ├── client9
            ├── client10
            ├── client11
            ├── client12
            ├── client13
            ├── client14
            ├── client15
            ├── client16
            ├── client17
            └── test1 (parameter: --data-dir ./Folder/test1)
    

Available FL Strategies
1. FedAvg (fedavg)
2. FedProx (fedprox)
3. FedOpt (fedopt)
4. FedRef (our proposed work.)
5. Adabest (adabest)
6. FedEve (fedeve)
7. A-FedPD (afedpd)

For example:

    #FedAvg
    python3 main.py -r 50 -e 3 -bs 256 -l 1e-5 -udp -1 -t femnist -m fedavg -NON 10 -g --degrade 0.0

    #FedProx
    python3 main.py -r 50 -e 3 -bs 256 -l 1e-5 -udp -1 -t femnist -m fedprox -NON 10 -g --degrade 0.0

    #FedRef
    python3 main.py -r 50 -e 3 -bs 256 -l 1e-5 -udp -1 -t femnist -m fedavg -NON 10 -g --degrade 0.0 --lda 1e-3 --rho 5