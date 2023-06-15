import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import click

from hominid_pipeline import hominid


@click.command()
@click.option("--config_file", type=str)
def main(config_file: str):

    save_path = config_file.split("config.yaml")[0]
    config = hominid.load_config(config_file)

    tuner = hominid.HominidTuner(
                                config,
                                epochs=100,
                                tuning_mode=False,
                                save_path=save_path,
                                dataset="deepstarr",
                                subsample=False
                                )

    # train model
    tuner.execute()


if __name__ == "__main__":
    main()
