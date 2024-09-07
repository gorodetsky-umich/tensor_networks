import argparse
import logging
import pathlib
import yaml

logger = logging.getLogger(__name__)

from load_config import load_yml_config, Config
import solver

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="input file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info('started')

    yml_dict = load_yml_config(pathlib.Path(args.input_file))
    settings = Config(**yml_dict)

    logger.info("settings:\n%r", settings)

    save_dir = pathlib.Path(settings.saving.directory)
    save_dir.mkdir(exist_ok=True)

    with open(save_dir / 'input.yaml', 'w') as f:
        yaml.dump(settings.model_dump(), f)

    # solver.check()

    logger.info('Finished')
