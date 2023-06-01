from pathlib import Path
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air import session
import tensorflow as tf
import yaml

import utils, model_zoo

class Hyperparameters:
    def __init__(self):
        self.parameters = {
            # Place all your hyperparameters here as key-value pairs
            "conv1_activation": tune.choice(["exponential", "relu"]), # --> use relu
            "conv1_batchnorm": tune.choice([True, False]), # use no bn
            "conv1_channel_weight": tune.choice(["softconv", "se", None]), # --> use either se or softconv
            "conv1_dropout": tune.choice([0.0, 0.1, 0.2, 0.3]),
            "conv1_filters": tune.choice([64, 96, 128, 256]), # --> use anything below 128
            "conv1_kernel_size": tune.choice([11, 15, 19]),
            "conv1_pool_type": tune.choice(["attention", "max_pool"]),
            "conv1_max_pool": tune.choice([0, 4, 8, 10, 20]),
            "conv1_attention_pool_size": tune.choice(range(40)),
            "conv1_type": tune.choice(["pw", "standard"]), # --> use pw
            "dense_activation": "relu",
            "dense_batchnorm": True,
            "dense_dropout": tune.choice([[0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]),
            "dense_units": tune.choice([[128, 128], [256, 128],[512, 256], [256, 256], [512, 512]]),
            "mha_d_model": tune.choice([96, 192]),
            "mha_dropout": tune.choice([0.0, 0.1, 0.2]),
            "mha_head_type": tune.choice(["pool", "task_specific"]),
            "mha_heads": tune.choice([4, 8]),
            "mha_layernorm": False,
            "output_activation": "linear",
            "output_shape": None
        }

    def get(self):
        return self.parameters

def load_config(config_path):
    config = yaml.full_load(open(config_path))
    return config

class HominidTuner:
    class DataProcessor:
        def __init__(self, subsample=False):
            self.subsample = subsample

        def load_data(self, data_split):
            return utils.load_deepstarr_data(data_split, subsample=self.subsample)

        @staticmethod
        def shape_info(x_data, y_data):
            N, L, A = x_data.shape
            output_shape = y_data.shape[-1]
            print(f"Input shape: {N, L, A}. Output shape: {output_shape}")
            return (L, A), output_shape

    class ModelBuilder:
        def __init__(self, config):
            self.config = config

        def build_model(self):
            print("Building model...")
            return model_zoo.base_model(**self.config)

    def __init__(self, config, epochs=60, tuning_mode=False, save_path=None, subsample=False):
        self.config = config
        self.subsample = subsample
        self.data_processor = self.DataProcessor(subsample=self.subsample)
        self.model_builder = self.ModelBuilder(config)
        self.epochs = epochs
        self.hyperparameters = Hyperparameters().get()
        self.tuning_mode = tuning_mode
        self.save_path = save_path


    def compile_and_train_model(self, model, x_train, y_train, x_valid, y_valid):

        model.compile(
            tf.keras.optimizers.Adam(lr=0.001),
            loss='mse',
            metrics=[utils.Spearman, utils.pearson_r]
        )
        model.summary()

        es_callback = self._get_early_stopping_callback()
        reduce_lr = self._get_reduce_lr_callback()

        callbacks = [es_callback, reduce_lr]

        if self.tuning_mode:
            tune_report_callback = self._get_tune_report_callback()
            callbacks.append(tune_report_callback)

        model.fit(
            x_train, y_train,
            epochs=self.epochs,
            batch_size=128,
            # verbose=0,
            shuffle=True,
            validation_data=(x_valid, y_valid),
            callbacks=callbacks
        )
        return model

    def update_config(self, key, value):
        self.config[key] = value
        self.model_builder = self.ModelBuilder(self.config)


    def get_evaluation_results(self):
        df = pd.read_csv(f'{self.save_path}/evaluation/model_performance.csv')
        return df

    def evaluate_model(self):

        print(f"Loading model and dataset!")

        x_test, y_test = self.data_processor.load_data("test")

        # Build the model
        model = self.model_builder.build_model()

        model.compile(
            tf.keras.optimizers.Adam(lr=0.001),
            loss='mse',
            metrics=[utils.Spearman, utils.pearson_r]
            )
        print(model.summary())
        model.load_weights(f'{self.save_path}/weights')

        # After training, you might want to evaluate your model:
        print(f"Evaluating model!")
        evaluation_path = f"{self.save_path}/evaluation"
        Path(evaluation_path).mkdir(parents=True, exist_ok=True)

        mse_dev, pcc_dev, scc_dev = utils.evaluate_model(model, x_test, y_test, "Dev")
        mse_hk, pcc_hk, scc_hk = utils.evaluate_model(model, x_test, y_test, "Hk")

        data = [{
            'MSE_dev':  mse_dev,
            'PCC_dev':  pcc_dev,
            'SCC_dev':  scc_dev,
            'MSE_hk':  mse_hk,
            'PCC_hk':  pcc_hk,
            'SCC_hk':  scc_hk,
        }]
        df = pd.DataFrame(data)
        pd.DataFrame(df).to_csv(f'{evaluation_path}/model_performance.csv', index=False)
        return df


    def _get_early_stopping_callback(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, verbose=1,
            mode='min', restore_best_weights=True
        )

    def _get_reduce_lr_callback(self):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, mode='min', verbose=1
        )

    def _get_tune_report_callback(self):
        return TuneReportCallback({"pearson_r": "pearson_r", "val_pearson_r": "val_pearson_r"})



    def execute(self):

        try:
            # Load the data
            x_train, y_train = self.data_processor.load_data("train")
            x_valid, y_valid = self.data_processor.load_data("valid")
            x_test, y_test = self.data_processor.load_data("test")

            # update config
            (L, A), output_shape = self.data_processor.shape_info(x_train, y_train)
            self.update_config("input_shape", (L, A))
            self.update_config("output_shape", output_shape)

            # Build the model
            model = self.model_builder.build_model()

            # Train the model
            model = self.compile_and_train_model(model, x_train, y_train, x_valid, y_valid)

            print("Done training the model!")

        except KeyboardInterrupt:
            print("Training interrupted")

        finally:
            # save model weights
            if self.tuning_mode:
                self.save_path = f'{Path(session.get_trial_dir())}'

            model.save_weights(f'{self.save_path}/weights')

            # save the model config
            with open(os.path.join(self.save_path, 'config.yaml'), 'w') as file:
                documents = yaml.dump(self.config, file)

            # After training, you might want to evaluate your model:
            print(f"Evaluating model!")
            evaluation_path = f"{self.save_path}/evaluation"
            Path(evaluation_path).mkdir(parents=True, exist_ok=True)

            mse_dev, pcc_dev, scc_dev = utils.evaluate_model(model, x_test, y_test, "Dev")
            mse_hk, pcc_hk, scc_hk = utils.evaluate_model(model, x_test, y_test, "Hk")

            data = [{
                'MSE_dev':  mse_dev,
                'PCC_dev':  pcc_dev,
                'SCC_dev':  scc_dev,
                'MSE_hk':  mse_hk,
                'PCC_hk':  pcc_hk,
                'SCC_hk':  scc_hk,
            }]
            df = pd.DataFrame(data)
            pd.DataFrame(df).to_csv(f'{evaluation_path}/model_performance.csv', index=False)



    def tune(self, num_training_iterations):
        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration", max_t=400, grace_period=20
        )

        param_space = self.hyperparameters

        tuner = tune.Tuner(
            tune.with_resources(self.execute, resources={"cpu": 4, "gpu": 1}),
            tune_config=tune.TuneConfig(
                metric="val_pearson_r",
                mode="max",
                scheduler=sched,
                num_samples=200,
            ),
            run_config=air.RunConfig(
                name="tune_hominid_v2",
                stop={
                    "val_pearson_r": 0.95,
                    "training_iteration": num_training_iterations
                },
                callbacks=[
                WandbLoggerCallback(
                    project="tune_hominid_v2",
                    log_config=True,
                    upload_checkpoints=True,)]
                ),
            param_space=param_space,
        )

        results = tuner.fit()

        return results

    def interpret_model(self, layer=2, plot_filters=True, from_saved=False):
        print(f"Loading model and dataset!")

        x_test, y_test = self.data_processor.load_data("test")

        # Build the model
        model = self.model_builder.build_model()

        model.compile(
            tf.keras.optimizers.Adam(lr=0.001),
            loss='mse',
            metrics=[utils.Spearman, utils.pearson_r]
            )
        print(model.summary())
        model.load_weights(f'{self.save_path}/weights')

        print(f"Interpreting filters!")
        threshold = 0.5
        window = 20

        W, sub_W, counts = utils.calculate_filter_activations(
                            model,
                            x_test,
                            self.save_path,
                            layer,
                            threshold,
                            window,
                            batch_size=64,
                            plot_filters=plot_filters,
                            from_saved=from_saved
                        )

#         utils.make_directory("/home/chandana/projects/hominid_pipeline/temp/tune_v2")
#             shutil.copy("-r", params_path, "/home/chandana/projects/hominid_pipeline/temp/tune_v2")

        print("Finished interpreting filters!")
        return
