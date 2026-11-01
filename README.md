# Machine Enabled TransfEction OptimizeR (METEOR)
===================================================================

## Setup and Installation

### Requirements
- Python 3.10
- CUDA 11.6
- PyTorch 1.13.1
- Anaconda 23.1.0

### Creating the Environment
1. Load the required modules and create a new Anaconda environment:
    ```bash
    conda create -n meteor_env python=3.10
    source activate meteor_env
    ```

2. Install dependencies:
    ```bash
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    pip install lmdb==1.4.0 ml-collections==0.1.1 numpy==1.23.4 scipy==1.9.3 tensorboardX==2.5.1 tqdm==4.64.1 tokenizers==0.13.2 pyprojroot==0.2.0 pandas==1.5.2 scikit-learn==1.2.0 rdkit-pypi==2022.9.3
    ```

3. Install Uni-Core, compatible with the specified versions:
    ```bash
    pip install https://github.com/dptech-corp/Uni-Core/releases/download/0.0.2/unicore-0.0.1+cu116torch1.13.1-cp310-cp310-linux_x86_64.whl
    ```

## Already Preprocessed Data
Data for some experiments have already been preprocessed and added to this repository. This includes:
- `data/processed_data_dirs/proc_random_split`: Dataset processed into eight folds, each with a random 20% test holdout
- `data/processed_data_dirs/proc_scaffold_split`: Dataset processed into eight folds, each with the same modulator scaffold test holdout
- `data/processed_data_dirs/proc_full`: Dataset processed as one standalone LMDB file, for full dataset inference only (not for training)

## Data Preprocessing
Preprocessing is done to make lmdb dataset (stored in `processed_data_dirs/`) from json files (stored in `data_json/`). Scripts for generating processed datasets from JSON files:
- `data/preprocess_data_for_infer.py`: Processes a set of JSON files into inferrable LMDB files.
    - `--path`: Path to the directory of JSON files
    - `--output_dir`: Path at which the processed LMDBs will be output
- `data/preprocess_data_for_training.py`: Includes CACO2 cell transfection data.
    - `--data_json_path`: Path to the individual training data JSON file (by default: `data/data_json/inhouse_lnp_booster_data_full_withLANCE_20240907.json`)
    - `--output_path`: Path to the processed LMDB output directory
    - `--kfold`: Number of train/test/valid folds output
    - `--test_ratio`: Size of random test split, input as a decimal (e.g. `0.2`)
    - `--scaffold_ids_json_path`: Path to list of separated modulator IDs used for scaffold/out-of-distribution testing (by default: `data/scaffold_ids.json`)
    - Note: only `--test_ratio` or `--scaffold_ids_json_path` should be set, not both

## Training
For training, run:
```
python train/train_grid_search.py --config <CONFIG_PATH>
```
There are two pre-made config files for the main training experiments:
- `train/configs/random.json`: Training on random 20% test holdout split
- `train/configs/scaffold.json`: Training on scaffold test holdout split

Each config is set to train on the `V0` fold by default, but this can be changed by altering the `root_task_name` config variable. Additionally, the config files are set by default to train five sets of weights, each across a different random seed.

## Inference
Run inference using pretrained models:
```
python infer/deploy_infer.py --config <CONFIG_PATH>
```

There is one pre-made config file at `infer/configs/default.json` that will run inference across the full processed dataset. Predictions are stored as .pkl files in the specified output directories. These files end with ".out.pkl" in their names. By default, the output directory is `infer/out`.

## Virtual Library Generation
Generate a virtual library of LNPs using:
```
python vlib/generators/parallelized_generator.py --config <CONFIG_PATH>
```

There are two pre-made virtual library generation configs:
- `vlib/configs/sample.yaml`: Creates a much smaller virtual library, intended for testing functionality and not in silico hit selection
- `vlib/configs/full.yaml`: Creates a much larger virtual library, spanning all component type combinations and varying molarities

## Key Files & Folders
- `data/data_json/`: Stores raw LNP data
- `data/processed_data_dirs/`: Stores processed LMDB datasets for training and inference
- `infer/`: Stores inference scripts and config files
- `train/`: Stores training scripts and config files
- `vlib/`: Stores virtual library generation scripts, configs, and full component lists
- `weights/`: Stores pretrained METEOR weights, accompanying inference script to use them is `infer/deploy_infer.py` ([CHANGE THIS LINK](https://drive.google.com/drive/folders/1IBz8iWrPX5Xnlb02VaTNR-7xuKuYUHZv?usp=drive_link))
- `unimol/`: Core logic for model operations.
- `ckp/`: Pretrained model weights ([download here](https://drive.google.com/drive/folders/1Ul89o6Vj93T01foKa1-898H32bJLTJkm?usp=drive_link))

### Note
Code has been test on Linux and installation is expected to take less than 2 hours on a typical computer with internet connection.