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


    # Load inputs
    yml_dict = load_yml_config(pathlib.Path(args.input_file), logger)
    config = Config(**yml_dict)

    # Get directory set up
    save_dir = pathlib.Path(config.saving.directory)
    save_dir.mkdir(exist_ok=True)
    with open(save_dir / 'input.yaml', 'w') as f:
        yaml.dump(config.model_dump(), f)

    logging.basicConfig(filename= save_dir/ 'log.log', level=logging.INFO)
    logger.info('started')
    logger.info("config:\n%r", config)

    # Start Create STUFF!!
    solver.main_loop(config, logger, save_dir)
    # solver.check()

    logger.info('Finished')
