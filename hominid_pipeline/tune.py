import click
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,6,7"
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from filelock import FileLock
import ray
from ray import air, tune
from ray.air import CheckpointConfig, session
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
import tensorflow as tf
from tensorflow import keras
import wandb

def load_data(data_path='/home/chandana/projects/hominid_pipeline/data/synthetic_dataset.h5'):

    with h5py.File(data_path, 'r') as dataset:
        x_train = np.array(dataset['X_train']).astype(np.float32)
        y_train = np.array(dataset['Y_train']).astype(np.float32)
        x_valid = np.array(dataset['X_valid']).astype(np.float32)
        y_valid = np.array(dataset['Y_valid']).astype(np.int32)
        x_test = np.array(dataset['X_test']).astype(np.float32)
        y_test = np.array(dataset['Y_test']).astype(np.int32)

    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def build_model(conv1_activation, conv1_filters, input_shape, output_shape):

    # l2 regularization
    l2 = keras.regularizers.l2(1e-6)

    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1 - convolution
    nn = keras.layers.Conv1D(
                        filters=conv1_filters,
                        kernel_size=19,
                        strides=1,
                        activation=None,
                        use_bias=False,
                        padding='same',
                        kernel_regularizer=l2)(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(conv1_activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    # layer 2 - convolution
    nn = keras.layers.Conv1D(
                        filters=128,
                        kernel_size=7,
                        strides=1,
                        activation=None,
                        use_bias=False,
                        padding='same',
                        kernel_regularizer=l2)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=25)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    # layer 3 - Fully-connected
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(
                        512,
                        activation=None,
                        use_bias=False,
                        kernel_regularizer=l2)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # Output layer
    logits = keras.layers.Dense(
                        output_shape,
                        activation='linear',
                        use_bias=True,
                        kernel_initializer='glorot_normal',
                        bias_initializer='zeros')(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def pipeline(config):

    # ==============================================================================
    # Load dataset
    # ==============================================================================

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data()

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

    model = build_model(**config)

    # set up optimizer and metrics
    auroc = keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[auroc, aupr]
            )

    return x_train, y_train, x_valid, y_valid, x_test, y_test, model

def train_fn(config: dict):

    x_train, y_train, x_valid, y_valid, x_test, y_test, model = pipeline(config)
    epochs = 5

    tune_trial_dir = Path(session.get_trial_dir())
    # early stopping callback
    es_callback = tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss',
                                        patience=10,
                                        verbose=1,
                                        mode='min',
                                        restore_best_weights=True
                                        )
    # reduce learning rate callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                                        monitor='val_loss',
                                        factor=0.2,
                                        patience=3,
                                        min_lr=1e-7,
                                        mode='min',
                                        verbose=1
                                        )
    # train model
    model.fit(
          x_train, y_train,
          epochs=epochs,
          batch_size=128,
          verbose=0,
          shuffle=True,
          validation_data=(x_valid, y_valid),
          callbacks=[
              es_callback,
              reduce_lr,
              TuneReportCallback({
                  "val_aupr": "val_aupr",
              }),
        ]
      )
    model.save_weights(f'{tune_trial_dir}/weights')

    return

def tune_model(num_samples):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(train_fn, resources={"cpu": 4, "gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="val_aupr",
            mode="max",
            scheduler=sched,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="test_tune",
            stop={
                "val_aupr": 0.95,
                "training_iteration": 100
            },
            callbacks=[
            WandbLoggerCallback(
                project="test_tune",
                log_config=True,
                upload_checkpoints=True,)]
            ),
        param_space={
            "conv1_activation": tune.choice(["exponential", "relu"]),
            "conv1_filters": tune.choice([64, 96, 128, 256]),
        },
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


@click.command()
@click.option("--smoke_test", type=bool)
def main(smoke_test: bool):


    tune_model(num_samples=5 if smoke_test else 100)


if __name__ == "__main__":
    main()
