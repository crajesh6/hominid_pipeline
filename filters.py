import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import click
import json
import shutil
import sys
from pathlib import Path
import pandas as pd
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
from ray.air import session
import tensorflow as tf
import yaml

import utils, model_zoo, hominid



@click.command()
@click.option("--config_file", type=str)
# @click.option("--layer", type=int)
def main(config_file: str):

    save_path = config_file.split("config.yaml")[0]
    config = hominid.load_config(config_file)

    tuner = hominid.HominidTuner(config, epochs=1, tuning_mode=False, save_path=save_path)

    # interpret model
    print("Evaluating first layer!")
    tuner.interpret_model(layer=3)

    print(f"Evaluating second layer!")
    if(config['conv1_channel_weight'] == "softconv"):
        layer = 5
    elif(config['conv1_channel_weight'] == "se"):
        layer = 8
    else:
        sys.exit()

    tuner.interpret_model(layer=layer)


if __name__ == "__main__":
    main()
