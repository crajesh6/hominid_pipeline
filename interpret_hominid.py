import h5py, time
import click
import importlib
import numpy as np
import pandas as pd
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pathlib import Path

from filelock import FileLock
import json

import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from scipy.stats import spearmanr
import sh
import wandb
import yaml

from tfomics import moana, impress

import model_zoo, utils




def interpret_model(params_path):

    print(f"Loading model and dataset!")

    config = json.load(open(f"{params_path}/params.json"))
    print(config)
    weights_path = f"{params_path}/weights"

    config = json.load(open(f"{params_path}/params.json"))
    _, _, _, _, x_test, y_test, model = utils.hominid_pipeline(config)

    model.compile(
        tf.keras.optimizers.Adam(lr=0.001),
        loss='mse',
        metrics=[utils.Spearman, utils.pearson_r]
        )
    print(model.summary())

    model.load_weights(f'{params_path}/weights')


    # # Evaluate model

    print(f"Evaluating model!")
    # run for each set and enhancer type
    evaluation_path = f"{params_path}/evaluation"
    Path(evaluation_path).mkdir(parents=True, exist_ok=True)

    mse_dev, pcc_dev, scc_dev = utils.evaluate_model(model, x_test,  y_test, "Dev")
    mse_hk, pcc_hk, scc_hk = utils.evaluate_model(model, x_test,  y_test, "Hk")

    data = [{
        'MSE_dev':  mse_dev,
        'PCC_dev':  pcc_dev,
        'SCC_dev':  scc_dev,
        'MSE_hk':  mse_hk,
        'PCC_hk':  pcc_hk,
        'SCC_hk':  scc_hk,
    }]
    df = pd.DataFrame(data)
    pd.DataFrame(df).to_csv(f'{evaluation_path}/model_performance.csv')

    # # Interpret filters
    # Usage example
    print(f"Interpreting filters!")
    layer = 2
    threshold = 0.5
    window = 20

    W, sub_W, counts = utils.calculate_filter_activations(
                        model,
                        x_test,
                        params_path,
                        layer,
                        threshold,
                        window,
                        batch_size=64,
                        plot_filters=True,
                        from_saved=False
                    )

    utils.make_directory("/home/chandana/projects/hominid_pipeline/temp/tune_v2")
    sh.cp("-r", params_path, "/home/chandana/projects/hominid_pipeline/temp/tune_v2")

    print("Finished interpreting filters!")
    return



@click.command()
@click.option("--params_path", type=str)

def main(params_path: str):
    interpret_model(params_path)


if __name__ == "__main__":
    main()
