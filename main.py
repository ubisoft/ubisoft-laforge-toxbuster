from loguru import logger

import argparse

from src.Benchmark.BenchmarkerConfig import BenchmarkerConfig
from src.Benchmark.Benchmarker import Benchmarker
from src.Infer.InfererConfig import InfererConfig
from src.Infer.Inferer import Inferer
from src.Training.Trainer import Trainer
from src.Training.TrainerConfig import TrainerConfig

@logger.catch
def main(mode, config: str, base_config: str, args: dict):
    '''
    Main script to train our model.

    Parameters:
    ------------
    mode: str
        One of ["Train", "Infer", "Benchmark"]
    config: str
        Config file to use for the respective mode. Medium priority. Overrides base_config.
    base_config: str
        Base config file to use for the respective mode. Least priority.
    args: dict
        Configs related to the respective mode. Highest priority. Overrides both config and base_config.
    '''

    # Create the list of config files. Note, priority is given to FIFO.
    config_files = []
    if (config is not None):
        config_files.append(config)
    if (base_config is not None):
        config_files.append(base_config)


    # Only keep configs we want to override. Default from argparse is None.
    config = {}
    for k,v in vars(args).items():
        if v is not None:
            config[k] = v
    configs = [config]
    configs.append({"data_output_init_args": {"random_state": args.datasplit_random_state}})

    if mode == "Train":
        config = TrainerConfig.from_configs_and_config_files(configs, config_files)
        trainer = Trainer(config)
        trainer.train()

    if mode == "Infer":
        config = InfererConfig.from_configs_and_config_files(configs, config_files)
        inferer = Inferer(config)
        inferer.infer()

    if mode == "Benchmark":
        config = BenchmarkerConfig.from_configs_and_config_files(configs, config_files)
        benchmarker = Benchmarker(config)
        benchmarker.benchmark()



def add_config_parser(subparsers, parent_parser: argparse.ArgumentParser, subcommand: str, config_class):
    '''
    Adds a new subparser with the provided subcommand and config_class.

    Using the Named Tuples field_defaults to grab all fields & use the type of the default.

    Parameter:
    -------------
    subparsers:argparse.ArgumentParser
        ArgumentParser
    parent_parser: argparse.ArgumentParser
        Parent parser for common arguments
    subcommand: str
        Name of the subcommand
    config_class: class of the config
        Classes like TrainerConfig, InfererConfig, BenchmarkerConfig
    '''
    subcommand_arg_parser = subparsers.add_parser(subcommand, help=f"{subcommand} help", parents=[parent_parser])
    for config_name, config_default_value in config_class._field_defaults.items():
        subcommand_arg_parser.add_argument(f"--{config_name}", type=type(config_default_value))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="MAIN")


    # Parent parser
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--config",
        help="Path to the config file",
        type=str,
    )
    parent_parser.add_argument(
        "--base_config",
        help="Path to the base config file. Least priority.",
        type=str,
    )
    parent_parser.add_argument(
        "--log_file",
        help="Path to store log file.",
        type=str,
        default="log.log"
    )

    parent_parser.add_argument(
        "--datasplit_random_state",
        help="Datasplit random state.",
        type=int,
        default=42
    )

    subparsers = parser.add_subparsers(help='sub-command help', dest='mode')

    add_config_parser(subparsers, parent_parser, "Train", TrainerConfig)
    add_config_parser(subparsers, parent_parser, "Infer", InfererConfig)
    add_config_parser(subparsers, parent_parser, "Benchmark", BenchmarkerConfig)

    args = parser.parse_args()
    logger.add(args.log_file)

    main(mode = args.mode, config = args.config, base_config = args.base_config, args=args)


