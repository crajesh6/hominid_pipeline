# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import click

import hominid_pipeline


@click.command()
@click.option("--config_file", type=str)
def main(config_file: str):

    save_path = config_file.split("config.yaml")[0]
    config = hominid_pipeline.load_config(config_file)

    tuner = hominid_pipeline.HominidTuner(
                                config,
                                epochs=100,
                                tuning_mode=False,
                                save_path=save_path
                                )

    # train model
    tuner.execute()


if __name__ == "__main__":
    main()
