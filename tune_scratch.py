import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4,5,6,7,8,9"
import h5py, time
import click
import importlib
import numpy as np
import pandas as pd
from pathlib import Path

from filelock import FileLock
import json
import ray
from ray import air, tune
from ray.air import CheckpointConfig
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.air import session
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
import tensorflow as tf
from tensorflow import keras
from scipy.stats import spearmanr
import wandb
import yaml

import model_zoo, utils

# set seed for reproducibility
np.random.seed(0)

def Spearman(y_true, y_pred):
     return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                       tf.cast(y_true, tf.float32)], Tout = tf.float32) )
from keras import backend as K

def pearson_r(y_true, y_pred):
    # use smoothing for not resulting in NaN values
    # pearson correlation coefficient
    # https://github.com/WenYanger/Keras_Metrics
    epsilon = 10e-5
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return K.mean(r)

def load_deepstarr_data(
        data_split: str,
        data_dir='/home/chandana/projects/hominid_pipeline/data/deepstarr_data.h5',
        subsample: bool = False
    ) -> (np.ndarray, np.ndarray):
    """Load dataset"""

    # load sequences and labels
    with FileLock(os.path.expanduser(f"{data_dir}.lock")):
        with h5py.File(data_dir, "r") as dataset:
            x = np.array(dataset[f'x_{data_split}']).astype(np.float32)
            y = np.array(dataset[f'y_{data_split}']).astype(np.float32).transpose()
    if subsample:
        if data_split == "train":
            x = x[:80000]
            y = y[:80000]
        elif data_split == "valid":
            x = x[:20000]
            y = y[:20000]
        else:
            x = x[:10000]
            y = y[:10000]
    return x, y

def hominid_pipeline(config):

    # ==============================================================================
    # Load dataset
    # ==============================================================================

    x_train, y_train = load_deepstarr_data("train", subsample=False)
    x_valid, y_valid = load_deepstarr_data("valid", subsample=False)
    x_test, y_test = load_deepstarr_data("test", subsample=False)

    N, L, A = x_train.shape
    output_shape = y_train.shape[-1]

    print(f"Input shape: {N, L, A}. Output shape: {output_shape}")

    config["input_shape"] = (L, A)
    config["output_shape"] = output_shape

    print(output_shape)

    # ==============================================================================
    # Build model
    # ==============================================================================

    print("Building model...")

    model = model_zoo.base_model(**config)

    return x_train, y_train, x_valid, y_valid, x_test, y_test, model

def tune_hominid(config: dict):

    x_train, y_train, x_valid, y_valid, x_test, y_test, model = hominid_pipeline(config)
    epochs = 60

    model.compile(
        tf.keras.optimizers.Adam(lr=0.001),
        loss='mse',
        metrics=[Spearman, pearson_r]
        )
    model.summary()

    # Write to the Tune trial directory, not the shared working dir
    tune_trial_dir = Path(session.get_trial_dir())

    # train model
    model.fit(
          x_train, y_train,
          epochs=epochs,
          batch_size=128,
          verbose=0,
          shuffle=True,
          validation_data=(x_valid, y_valid),
          callbacks=[
              TuneReportCallback({
                  "pearson_r": "pearson_r",
                  "val_pearson_r": "val_pearson_r",
              }),
        ]
      )
    model.save_weights(f'{tune_trial_dir}/weights')

    return

def tune_mnist(num_training_iterations):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(tune_hominid, resources={"cpu": 4, "gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="val_pearson_r",
            mode="max",
            scheduler=sched,
            num_samples=200,
        ),
        run_config=air.RunConfig(
            name="tune_hominid_pipeline-test",
            stop={
                "val_pearson_r": 0.95,
                "training_iteration": num_training_iterations
            },
            callbacks=[
            WandbLoggerCallback(
                project="raytune-hominid_pipeline-test",
                log_config=True,
                upload_checkpoints=True,)]
            ),
        param_space={
            "conv1_activation": tune.choice(["relu"]),                     # activation on 1st layer conv
            "conv1_batchnorm": tune.choice([False]),                       # batchnorm on 1st layer conv
            "conv1_channel_weight": tune.choice(["softconv", "se", None]), # soft attention on channels (1st layer conv)
            "conv1_dropout": 0.2,
            "conv1_filters": tune.choice([64, 96, 128, 256, 512]),
            "conv1_kernel_size": tune.choice([15, 19]),
            "conv1_pool_type": tune.choice(["attention", "max_pool"]),
            "conv1_max_pool": tune.choice([4, 8, 10, 20]),                 # if conv1 pool = max pool
            "conv1_attention_pool_size": tune.choice(range(40)),           # if conv1 pool = attention pool
            "conv1_type": tune.choice(["pw", "standard"]),                 # additive vs pairwise 1st conv layer
            "dense_activation": "relu",
            "dense_batchnorm": True,
            "dense_dropout": tune.choice([[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]),
            "dense_units": tune.choice([[128, 128], [256, 128],[512, 256], [256, 256], [512, 512],[1024, 512]]),
            "mha_d_model": tune.choice([96, 192]),
            "mha_dropout": 0.1,
            "mha_head_type": tune.choice(["pool", "task_specific"]),       # shared vs task specific atttention
            "mha_heads": tune.choice([4, 8]),
            "mha_layernorm": False,
            "output_activation": "linear",
            "output_shape": None
        },
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args, _ = parser.parse_known_args()

    tune_mnist(num_training_iterations=100 if args.smoke_test else 100)
