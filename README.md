# Scout: Predictive ML model maintenance
This repository contains the implementation for the paper "Scout: Predictive ML model maintenance". The code is divided among the datasets that it is evaluated on, Census, CIFAR10/CIFAR10-C, and Query Optimizer. 

## Installation
To download and install the required packages, run the requirements.txt file
```
pip3 install -r requirements.txt
```
Next, the datasets need to be cleaned, organized, and split into drifted and non-drifted partitions as described in the paper.

For the Census dataset run
```
python3 census_setup.py
```
The dataset used is available under census/data/adults.csv, which is used to create the drifted and non-drifted sets. It then creates four simulation patterns with the data: abrupt(1), gradual(2), periodic(3), and no drift(4). For each drift pattern, we create a simulation file named future_(num).csv, where the value of num determines the drift pattern.

To set up CIFAR 
- **Download the CIFAR-10C dataset**
  - Go to `https://zenodo.org/records/2535967` and download the `CIFAR10C` dataset.

- **Create the target directory**
  - Under the directory './cifar10/data/ create a folder `CIFAR-10-C`.
  - Extract the downloaded CIFAR10c files into the newly created directory `CIFAR-10-C`.

Now that we have the "drifted" data, we can set up similar simulations to the Census dataset by running:
```
python3 cifar_setup.py
```
First time running this will download the required CIFAR10 dataset and extract its contents. It will then train a ResNet18 classifier and create different simulations with varying drift patterns. Initial setup time for CIFAR is high and requires a GPU with at least 48 GB of VRAM; this is only for the initial setup.

## Training Scout metamodel 

Next, we train the proposed Scout metamodel and RiskAdvisor from Lahoti et al.(Responsible model deployment via model-agnostic uncertainty learning)
- Census: ``` python3 census_sense.py ```
- CIFAR: ``` python3 cifar_sense.py ```

These train the scout metamodel and the Risk Advisor model using discretized loss and mispredictions, respectively. It also generates the different datasets needed to train the metamodel and Risk Advisor, placing them under './census/data/' and './cifar10/data/'. 

## Running the simulation
5 different kinds of model maintenance techniques can be run on the different drift patterns created in the previous step. The different parameters that can be adjusted to vary the simulation are: 
- delta: Decides the ground truth delay in hours, delta=6 means ground truth is available 6 hours after the input query arrives.
- sc: Selects the different drift patterns to run, 1=Abrupt, 2=Gradual, 3=Periodic, and 4=No Drift
- kappa: User-defined threshold to trigger retraining, number of tolerable high losses/mispredictions above the baseline
- beta: User-defined threshold for the percentage of high-loss/misprediction ground truth required before retraining. E.g., a kappa=6000 and beta=0.5 would wait for 0.5x6000 or 3000 high-loss data ground truth to be available before retraining
- retraining_period: Specific to periodic techniques, a fixed interval of time after the previous retraining/start when retraining is performed again.
- reactive(True/False): Specific to Scout(proposed), changes between reactive and predictive scout

- Naive reactive: Waits for ground truth to evaluate the ML model performance before retraining.
  - Census: ``` python3 census_reactive.py <delta> <sc> <kappa>```
  - CIFAR: ``` python3 cifar_reactive.py <delta> <sc> <kappa>```
 
- Periodic retraining: Retrains after a fixed interval of time regardless of ML model performance.
  - Census: ``` python3 census_periodic.py <delta> <sc> <retraining_period>```
  - CIFAR: ``` python3 cifar_periodic.py <delta> <sc> <retraining_period>```

- DDLA: Uses the decision tree classifier technique from "Efficiently Mitigating the Impact of Data Drift on Machine Learning Pipelines" to predict ML model mispredictions without ground truth.
  - Census: ``` python3 census_ddla.py <delta> <sc> <kappa> <beta>```
  - CIFAR: ``` python3 cifar_ddla.py <delta> <sc> <kappa> <beta>```

- Risk Advisor: Uses an ensemble metamodel technique from "Responsible model deployment via model-agnostic uncertainty learning" by Lahoti et al, to predict ML model mispredictions without ground truth.
  - Census: ``` python3 census_risk_advisor.py <delta> <sc> <kappa> <beta>```
  - CIFAR: ``` python3 cifar_risk_advisor.py <delta> <sc> <kappa> <beta>```
 
- Scout: Uses the decision tree classifier technique from "Efficiently Mitigating the Impact of Data Drift on Machine Learning Pipelines" to predict ML model mispredictions without ground truth.
  - Census: ``` python3 census_sim_fbts_paper.py <delta> <sc> <kappa> <beta> <reactive>```
  - CIFAR: ``` python3 cifar_sim_fbts_paper.py <delta> <sc> <kappa> <beta> <reactive>```
 
The various simulation results are saved in './census/data/' or './cifar10/data/' with the chosen parameters in the filename. Each simulation produces two files: a .csv file with all the details of each input and a .npy file, which stores the different retraining characteristics. 
TODO: Add batch files to run different simulations with parameters simultaneously on a cluster

