"""Command Line Interface."""

import argparse
import os

import ganji.project


def _args_to_config(args) -> ganji.project.Config:
    return ganji.project.Config(
        batch_size=args.batch_size,
        codepoint_set=args.codepoint_set,
        epoch_end=args.epoch_end,
        font=args.font,
        font_index=args.font_index,
        unit=args.unit,
        thickness_quantile_min=args.thickness_quantile_min,
        thickness_quantile_max=args.thickness_quantile_max,
    )


def add_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-B", "--batch-size", type=int, help="batch_size", default=1024)
    parser.add_argument(
        "-c", "--codepoint-set", help="codepoint set (kanji|jouyou-kanji|hiragana) [default: kanji]", default="kanji",
    )
    parser.add_argument("-E", "--epoch-end", type=int, help="epoch end", default=10000)
    parser.add_argument("-F", "--font", help="font file", required=True)
    parser.add_argument("-I", "--font-index", type=int, help="font index [default: 0]", default=0)
    parser.add_argument("-N", "--unit", type=int, help="(size / 4) [default: 10]", default=10)
    parser.add_argument(
        "-T", "--thickness-quantile-max", type=float, help="quantile of maximum thickness", default=None
    )
    parser.add_argument(
        "-t", "--thickness-quantile-min", type=float, help="quantile of minimum thickness", default=None
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
        import ganji.dnn.dcgan  # deferred

        dir = os.getcwd() if args.directory is None else args.directory
        ganji.dnn.dcgan.train(dir)

    train_parser = subparsers.add_parser("train", help="train generator and discriminator")
    train_parser.set_defaults(handler=command_train)

    def command_generate(args):
        import ganji.dnn.dcgan  # deferred

        dir = os.getcwd() if args.directory is None else args.directory
        ganji.dnn.dcgan.generate(dir, epoch=args.epoch, nice=args.nice)

    generate_parser = subparsers.add_parser("generate", help="generate output")
    generate_parser.add_argument("-e", "--epoch", type=int, help="specify epoch")
    generate_parser.add_argument("-n", "--nice", help="generate nicer output", default=False, action="store_true")
    generate_parser.set_defaults(handler=command_generate)

    def command_log(args):
        import ganji.dnn.dcgan  # deferred

        dir = os.getcwd() if args.directory is None else args.directory
        ganji.dnn.dcgan.log(dir)

    log_parser = subparsers.add_parser("log", help="show logs")
    log_parser.set_defaults(handler=command_log)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
