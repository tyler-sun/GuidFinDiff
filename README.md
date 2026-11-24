# GuidFinDiff: Guided Tabular Diffusion-based Method for Risk-aware Financial Data Generation
This repo details the code for GuidFinDiff, a tabular diffusion framework for synthesizing minority class samples using underlying feature patterns, which we classify as 'risk', rather than broad class labels. We use diffusion architectures for generating tabular data by conditioning at specific risk values and/or ranges rather than a class label for more precise samples in financial domains.

## TabDDPM: Modelling Tabular Data with Diffusion Models
This research builds upon TabDDPM, an earlier diffusion model for generating tabular data. To read about TabDDPM, their official report "TabDDPM: Modelling Tabular Data with Diffusion Models" can be found here: ([paper](https://arxiv.org/abs/2209.15421))

<!-- ## Results
You can view all the results and build your own tables with this [notebook](notebooks/Reports.ipynb). -->

## Setup the environment
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) (just to manage the env).
2. Run the following commands
    ```bash
    export REPO_DIR=/path/to/the/code
    cd $REPO_DIR

    conda create -n tddpm python=3.9.7
    conda activate tddpm

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt

    # if the following commands do not succeed, update conda
    conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
    conda env config vars set PROJECT_DIR=${REPO_DIR}

    conda deactivate
    conda activate tddpm
    ```

 ## Running the experiments

<!-- Here we describe the neccesary info for reproducing the experimental results.  
Use `agg_results.ipynb` to print results for all dataset and all methods. -->

### Datasets

Datasets used in the paper with our train/val/test splits, as well as datasets provided with the TabDDPM repo, are included (links below). The sources of the datasets used for GuidFinDiff are also listed in the paper appendix. 

Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
IEEE-CIS Fraud Detection: https://www.kaggle.com/competitions/ieee-fraud-detection/data 

- `scripts/csv_to_npy.py` can be used to split CSV datasets such as the Credit Card dataset into .npy partitions.

```bash
python scripts/csv_to_npy.py [path/to/dataset] [column for y-labels] [output/directory] --train_split [1 to 99] --val_split [1 to 99] --same_rate
```

Other datasets, including those from TabDDPM, can be loaded with the following commands:

``` bash
conda activate tddpm
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar
tar -xvf data.tar
```

### File structure
`tab-ddpm/` -- implementation of the proposed method  
`tuned_models/` -- tuned hyperparameters of evaluation model (CatBoost or MLP)

All main scripts are in `scripts/` folder:

- `scripts/pipeline.py` are used to train, sample and eval TabDDPM using a given config  
- `scripts/tune_ddpm.py` -- tune hyperparameters of TabDDPM
- `scripts/eval_[catboost|xgboost|mlp|simple].py` -- evaluate synthetic data using a tuned evaluation model or simple models
- `scripts/eval_seeds.py` -- eval using multiple sampling and multuple eval seeds
- `scripts/eval_seeds_simple.py` --  eval using multiple sampling and multuple eval seeds (for simple models)
- `scripts/tune_evaluation_model.py` -- tune hyperparameters of eval model (CatBoost or MLP)
- `scripts/resample_privacy.py` -- privacy calculation
- `scripts/find_risk.py` -- calculate risk from a provided dataset using logistic regression or CatBoost classifier
- `scripts/limit.py` and `scripts/filter_1_class.py` -- used in preliminary tests to find results on rudimentary forms of minority augmentation (aka generate samples normally, then filter for the first x number of samples of a specific class). Not needed for running TabDDPM or GuidFinDiff.

Experiments folder (`exp/`):
- All results and synthetic data are stored in `exp/[ds_name]/[exp_name]/` folder
- `exp/[ds_name]/config.toml` is a base config for tuning TabDDPM
- `exp/[ds_name]/eval_[catboost|mlp].json` stores results of evaluation (`scripts/eval_seeds.py`)  

`CONFIG_DESCRIPTION.md` contains a description of important parameters and structures of 'config.toml' including those needed for conditional data synthesis.

Baselines:
- `smote/`
- `CTGAN/` -- TVAE [official repo](https://github.com/sdv-dev/CTGAN)
- `CTAB-GAN/` --  [official repo](https://github.com/Team-TUD/CTAB-GAN)
- `CTAB-GAN-Plus/` -- [official repo](https://github.com/Team-TUD/CTAB-GAN-Plus)

### Examples from TabDDPM

One can run the model without risk conditioning, which can be used to generate samples using class labels for conditioning, or no conditioning at all.

<ins>Run TabDDPM tuning.</ins>   

Template and example (`--eval_seeds` is optional): 
```bash
python scripts/tune_ddpm.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds
python scripts/tune_ddpm.py churn2 6500 synthetic catboost ddpm_tune --eval_seeds
```

<ins>Run TabDDPM pipeline.</ins>   

Template and example  (`--train`, `--sample`, `--eval` are optional): 
```bash
python scripts/pipeline.py --config [path_to_your_config] --train --sample --eval
python scripts/pipeline.py --config exp/churn2/ddpm_cb_best/config.toml --train --sample
```
It takes approximately 7min to run the script above (NVIDIA GeForce RTX 2080 Ti).  

<ins>Run evaluation over seeds</ins>   
Before running evaluation, you have to train the model with the given hyperparameters (the example above).  

Template and example: 
```bash
python scripts/eval_seeds.py --config [path_to_your_config] [n_eval_seeds] [ddpm|smote|ctabgan|ctabgan-plus|tvae] synthetic [catboost|mlp] [n_sample_seeds]
python scripts/eval_seeds.py --config exp/churn2/ddpm_cb_best/config.toml 10 ddpm synthetic catboost 5
```

### Run risk conditioned synthesis

<ins>Find model hyperparameters via TabDDPM tuning.</ins>   

Template and example (`--eval_seeds` is optional): 
```bash
python scripts/tune_ddpm.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds
python scripts/tune_ddpm.py churn2 6500 synthetic catboost ddpm_tune --eval_seeds
```

<ins>Run the risk model for a given dataset using CatBoost (cb) or logistic regression.</ins>

```bash
python scripts/find_risk.py [path/to/dataset] --plot --model [cb|logistic regression]
```
To plot the risk distributions only from the synthetic data:
```bash
python scripts/find_risk.py [path/to/dataset] --plot_only
```
This will produce r_train, r_test, and r_val (if the split is enabled) .npy files which can be used for risk conditioning through risk_conditioning = true in the configuration file.

<ins>From there, set up the config file and trigger the pipeline:</ins>

```bash
python scripts/pipeline.py --config [path/to/config] --train --sample
python scripts/pipeline.py --config exp/credit-card/ddpm_tune_best/config.toml --train --sample

python scripts/eval_seeds.py --config [path/to/config] [n_eval_seeds] ddpm merged [catboost|mlp|xgboost] 1 --change_val 
python scripts/eval_seeds.py --config exp/credit-card/ddpm_tune_best/config.toml 10 ddpm merged xgboost 1 --change_val

```

The config file and synthesized data are usually expected to be in the exp(eriments) folder. Note the evaluation model should be different from model_type chosen in the config file. More details on configuration parameters can be found in `CONFIG_DESCRIPTION.md`.