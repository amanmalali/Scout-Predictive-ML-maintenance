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

