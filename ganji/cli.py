"""Command Line Interface."""

import argparse
import os
from typing import Optional

import ganji.project


def _select_implementation(mode: Optional[str]):
    if mode is None:
        import ganji.nn.dcgan

        return ganji.nn.dcgan
    mode = mode.lower().replace("-", "_")
    if mode == "dcgan":
        import ganji.nn.dcgan

        return ganji.nn.dcgan
    elif mode == "wgan":
        import ganji.nn.wgan

        return ganji.nn.wgan
    else:
        raise ValueError(f"invalid type: {mode}")


def _args_to_config(args) -> ganji.project.Config:
    return ganji.project.Config(
        mode=args.mode,
        batch_size=args.batch_size,
        codepoint_set=args.codepoint_set,
        epoch_end=args.epoch_end,
        font=args.font,
        font_index=args.font_index,
        unit=args.unit,
        density_quantile_min=args.density_quantile_min,
        density_quantile_max=args.density_quantile_max,
        dataset_random_seed=args.dataset_random_seed,
    )


def add_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-B", "--batch-size", type=int, help="batch_size", default=1024)
    parser.add_argument(
        "-c", "--codepoint-set", help="codepoint set (kanji|joyo-kanji|hiragana) [default: kanji]", default="kanji",
    )
    parser.add_argument("-D", "--density-quantile-max", type=float, help="quantile of maximum density", default=None)
    parser.add_argument("-d", "--density-quantile-min", type=float, help="quantile of minimum density", default=None)
    parser.add_argument("-E", "--epoch-end", type=int, help="epoch end", default=100000)
    parser.add_argument("-F", "--font", help="font file", required=True)
    parser.add_argument("-I", "--font-index", type=int, help="font index [default: 0]", default=0)
    parser.add_argument("-m", "--mode", help="project type (dcgan|wgan)", choices=["dcgan", "wgan"])
    parser.add_argument("-N", "--unit", type=int, help="(size / 4) [default: 10]", default=10)
    parser.add_argument(
        "--dataset-random-seed", type=int, help="seed for PRNG to generate a dataset [default: 0]", default=0
    )


def main():
    parser = argparse.ArgumentParser(description="This kanji does not exist.")
    parser.add_argument("-C", "--directory", help="project directory")

    subparsers = parser.add_subparsers()

    def command_new(args):
        ganji.project.new(args.dir, _args_to_config(args))

    new_parser = subparsers.add_parser("new", help="create project directory")
    add_init_arguments(new_parser)
    new_parser.add_argument("dir", help="project directory")
    new_parser.set_defaults(handler=command_new)

    def command_init(args):
        dir = os.getcwd() if args.directory is None else args.directory
        ganji.project.init(dir, _args_to_config(args))

    init_parser = subparsers.add_parser("init", help="initialize project directory")
    add_init_arguments(init_parser)
    init_parser.set_defaults(handler=command_init)

    def command_train(args):
        dir = os.getcwd() if args.directory is None else args.directory
        config, _state = ganji.project.load_metadata(dir)
        _select_implementation(config.mode).train(dir)

    train_parser = subparsers.add_parser("train", help="train generator and discriminator")
    train_parser.set_defaults(handler=command_train)

    def command_generate(args):
        dir = os.getcwd() if args.directory is None else args.directory
        config, _state = ganji.project.load_metadata(dir)
        _select_implementation(config.mode).generate(
            dir, rows=args.rows, columns=args.columns, epoch=args.epoch, seed=args.seed
        )

    generate_parser = subparsers.add_parser("generate", help="generate output")
    generate_parser.add_argument("-c", "--columns", type=int, help="the number of rows of an output image")
    generate_parser.add_argument("-e", "--epoch", type=int, help="specify epoch")
    generate_parser.add_argument("-r", "--rows", type=int, help="the number of rows of an output image")
    generate_parser.add_argument("-s", "--seed", type=int, help="seed for PRNG to generate an output image")
    generate_parser.set_defaults(handler=command_generate)

    def command_log(args):
        dir = os.getcwd() if args.directory is None else args.directory
        config, _state = ganji.project.load_metadata(dir)
        _select_implementation(config.mode).log(dir)

    log_parser = subparsers.add_parser("log", help="show logs")
    log_parser.set_defaults(handler=command_log)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
