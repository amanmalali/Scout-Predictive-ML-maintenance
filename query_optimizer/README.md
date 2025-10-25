# Scout for query optimizer dataser
This folder contains the implementation of Scout and other maintenance techniques for the query optimizer dataset

## Installation
To download and install the required packages, run the requirements.txt file
```
pip3 install -r query_requirements.txt
```
Next, the datasets need to be cleaned, organized, and split into drifted and non-drifted partitions as described in the paper.

To set up Query Optimizer dataset
- **Download the dataset and train the regression model**
  - Run ```python3 train_tpch.py``` to download the dataset and train the model

- **Generate loss data and drift patterns**
  - Run ```python3 gen_loss_tpch.py``` to generate loss values for different data splits
  - Next, run ```python3 gen_tpch_sim.py``` to generate the different different drift patterns and generate training data for the Scout metamodel


## Training Scout metamodel 

Next, we train the proposed Scout metamodel 
- Run ```python3 tpch_sense.py```
- To train the different temperature scaling methods run ```python3 tpch_train_temp_fbts.py```

## Running the simulation
5 different kinds of model maintenance techniques can be run on the different drift patterns created in the previous step. The different parameters that can be adjusted to vary the simulation are: 
- delta: Decides the ground truth delay in hours, delta=6 means ground truth is available 6 hours after the input query arrives.
- sc: Selects the different drift patterns to run, 1=Abrupt, 2=Gradual, 3=Periodic, and 4=No Drift
- kappa: User-defined threshold to trigger retraining, number of tolerable high losses/mispredictions above the baseline
- beta: User-defined threshold for the percentage of high-loss/misprediction ground truth required before retraining. E.g., a kappa=6000 and beta=0.5 would wait for 0.5x6000 or 3000 high-loss data ground truth to be available before retraining
- retraining_period: Specific to periodic techniques, a fixed interval of time after the previous retraining/start when retraining is performed again.
- reactive(True/False): Specific to Scout(proposed), changes between reactive and predictive scout

- Naive reactive: Waits for ground truth to evaluate the ML model performance before retraining.
  - Run: ``` python3 tpch_reactive.py <delta> <sc> <kappa>```

- Periodic retraining: Retrains after a fixed interval of time regardless of ML model performance.
  - Run: ``` python3 tpch_periodic.py <delta> <sc> <retraining_period>```


- Scout: Uses the decision tree classifier technique from "Efficiently Mitigating the Impact of Data Drift on Machine Learning Pipelines" to predict ML model mispredictions without ground truth.
  - Run: ``` python3 tpch_fbts_paper.py <delta> <sc> <kappa> <beta> <reactive>```

 
The various simulation results are saved in './data/' with the chosen parameters in the filename. Each simulation produces two files: a .csv file with all the details of each input and a .npy file, which stores the different retraining characteristics. 
TODO: Add batch files to run different simulations with parameters simultaneously on a cluster

