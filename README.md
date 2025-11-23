## Fed-Ref: Communication-Efficient Bayesian Fine Tuning with Reference Model

### Abstract
Federated learning (FL) collaboratively trains artificial intelligence (AI) models to ensure user data privacy. Sharing only model updates generated from local training using local client data with the server enhances user data privacy. However, model performance may suffer due to data and system heterogeneity among clients in FL scenarios. Previous studies have proposed model optimization, fine-tuning, or personalization to achieve optimal performance. Despite these efforts, models resulting from FL scenarios exhibit catastrophic forgetting, which increases clients' communication and computing costs for model optimization and increases energy consumption. To address these challenges, we propose a reference model-based fine-tuning method for federated learning that overcomes catastrophic forgetting in each round. Our method is derived from Bayesian parameter-efficient transfer learning and includes an optimal proximal term. It utilizes a reference model that incorporates previous model parameters and reviews previous global features in the model optimization step to decrease the catastrophic forgetting issue. As a result, our method achieves higher model performance and lower communication and computing costs for clients than existing works.

Paper link: https://doi.org/10.48550/arXiv.2506.23210

### Introduction
Recently, in the AI industry, federated learning (FL) has been proposed as a prominent solution to protect users' data privacy while allowing collaborative model training between independent affiliations. Users' data are fundamentally protected because clients share only updated local model parameters (resulting from local training using their local data) and other non-private values with the server or other clients. The server then computes the global model by aggregating the local models received by the clients. However, as highlighted in \cite{fedavg}, FL solutions face numerous challenges. Our focus is on optimizing model performance and decreasing computing costs on client devices.
    
    1. Model performance optimization in FL scenarios.
    2. Decreasing computing cost and energy consumption.
    3. How to protect the model from malicious users.

<img src="./res/FedRef.png" alt="Basic FL system" width="700"/>

We introduce our FedRef: a communication-efficient Bayesian fine-tuning approach with a reference model. This concept overcomes catastrophic forgetting in each round by integrating previous model features into the maximum a posteriori (MAP) problem.

### FedRef: Communication-Efficient Bayesian Fine Tuning with Reference Model
For optimal model performance and low client computing cost, we proposed FedRef: communication-efficient Bayesian fine-tuning with reference model which overcome catastrophic forgetting by inferring of previous rounds model. We carefully defined the model proximal term as a MAP problem for FL scenarios in under equation derived from the Bayesian fine-tuning paper.

<img src="./res/equation1.png" alt="MAP problem" width="500"/>

Selected client numbers $k \in [1,2,3,..,K]$, where $K$ is the total number of selected clients. $D_{ref}$: represents synthetic data of a reference model that incorporates features from previous rounds. $D_{ref}$ is defined solely for explaining our optimal MAP problem likes under equation. This concept enables overcoming catastrophic forgetting in each round by integrating previous round features into the MAP problem and then optimizing the integrated MAP value. Finally, the objective function can be expressed by combining equations to form under equation.

<img src="./res/equation2.png" alt="Bayesian" width="500"/>

In the above equation, the constant value $\sum_{k}^{K} F_k$ denotes the sum of client losses. The diagonal matrix $\mathrm{diag}(W_1, \dots, W_K)$ represents aggregation weights (e.g. $\frac{n_i}{n}$). $\sum_{i} (\theta_i - \theta_{0,i})^2$ signifies $L_2$ regularization of model $\theta-\theta_0$. Regarding parameter requirements, only clients' losses are needed for our FedRef concept. Only on the server side, model optimization is performed, which can decrease client computing cost. In the FedRef, $\theta^2$ should be set as the reference model $\theta_{ref}$. For detail on $\theta_{ref}$, the reference model is defined as:

<img src="./res/referenceModel.png" alt="Basic FL system" width="350"/>

<img src="./res/equation3.png" alt="reference model" width="300"/>

where $p$ is the number of selected subset of previous aggregation model which can be set heuristically and function $\text{A}$, which represents previous rounds global feature, is calculated as

<img src="./res/equation4.png" alt="reference model detail" width="300"/>

Experimentally, a suitable value for $p$ was analyzed heuristically to be 3 to 5, but it should be considered about memory resources for saving parameters.

### Settings
| Environment set  | Settings for detail                             |
|------------------|-------------------------------------------------|
| FL framework     | Flower: a friendly federated learning framework |
| Language         | Python                                          |
| Operation System | Linux 24.04 LTS                                 |
| GPU              | Nvidia RTX 4090                                 |
| Tools            | Visual studio code                              |

### Results
FEMNIST

<img src="./res/femnist/loss.png" alt="reference model detail" width="250"/><img src="./res/femnist/acc.png" alt="reference model detail" width="250"/><img src="./res/femnist/f1score.png" alt="reference model detail" width="250"/>

CINIC10

<img src="./res/cinic10/loss.png" alt="reference model detail" width="250"/><img src="./res/cinic10/acc.png" alt="reference model detail" width="250"/><img src="./res/cinic10/f1score.png" alt="reference model detail" width="250"/>
### Run

Available FL Strategies
1. FedAvg
2. FedProx
3. FedOpt
4. FedRef 
(our proposed work.)

! python main.py --version "AnyName" --type " fets|femnist|cinic10 " --seed "AnyNumber"--round "Round-Number" --epoch "LocalTrainingEpoch-Number" --batch-size "fets:1 |brats:1 | octdl:ProperNumber"--data-dir "TestData-Folder" --client-dir "Train-Client-Partitions-Folder" --lr "LearningRate" --mode "fedavg|fedprox|fedopt|fedref"--client-num "total number of clients" --gpu True| False --result-path "path" --prime "number of reference model" --lda "Bayesian aggregate learning rate"

for example:

    python main.py -v FedRefFEMNIST --data-dir None -cd None -r 30 -bs 256 -m "fedref" -t "femnist" --client-num 10 --epoch 3 --lr 0.000009 --lda 0.001 --prime 3

    python main.py -v FedRefFeTs --data-dir "folder1" -cd "folder2" -r 30 -bs 1 -m "fedref" -t "fets" --client-num 10 --epoch 3 --lr 0.000009 --lda 0.001 --prime 3