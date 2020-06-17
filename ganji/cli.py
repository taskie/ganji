"""Command Line Interface."""

import argparse
import os

import ganji.project


def _args_to_dict(args):
    return {
        "props": {
            "batch_size": args.batch_size,
            "codepoint_set": args.codepoint_set,
            "epoch_end": args.epoch_end,
            "font": args.font,
            "unit": args.unit,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="freetype loader")
    parser.add_argument("-C", "--directory", help="project directory")

    subparsers = parser.add_subparsers()

    def command_new(args):
        ganji.project.new(args.dir, _args_to_dict(args))

    new_parser = subparsers.add_parser("new", help="create project directory")
    new_parser.add_argument("-B", "--batch-size", type=int, help="batch_size", default=1024)
    new_parser.add_argument(
        "-c", "--codepoint-set", help="codepoint set (kanji|jouyou-kanji|hiragana) [default: kanji]", default="kanji",
    )
    new_parser.add_argument("-E", "--epoch-end", type=int, help="epoch end", default=10000)
    new_parser.add_argument("-F", "--font", help="font file", required=True)
    new_parser.add_argument("-N", "--unit", type=int, help="(size / 4) [default: 10]", default=10)
    new_parser.add_argument("DIR", help="project directory")
    new_parser.set_defaults(handler=command_new)

    def command_init(args):
        dir = os.getcwd() if args.directory is None else args.directory
        ganji.project.init(dir, _args_to_dict(args))

    init_parser = subparsers.add_parser("init", help="initialize project directory")
    init_parser.add_argument("-B", "--batch-size", type=int, help="batch_size", default=1024)
    init_parser.add_argument(
        "-c", "--codepoint-set", help="codepoint set (kanji|jouyou-kanji|hiragana) [default: kanji]", default="kanji",
    )
    init_parser.add_argument("-E", "--epoch-end", type=int, help="epoch end", default=10000)
    init_parser.add_argument("-F", "--font", help="font file", required=True)
    init_parser.add_argument("-N", "--unit", type=int, help="(size / 4) [default: 10]", default=10)
    init_parser.set_defaults(handler=command_init)

    def command_train(args):
        import ganji.gan  # deferred

        dir = os.getcwd() if args.directory is None else args.directory
        ganji.gan.train(dir)

    train_parser = subparsers.add_parser("train", help="train generator and discriminator")
    train_parser.set_defaults(handler=command_train)

    def command_generate(args):
        import ganji.gan  # deferred

        dir = os.getcwd() if args.directory is None else args.directory
        ganji.gan.generate(dir, epoch=args.epoch, nice=args.nice)

    generate_parser = subparsers.add_parser("generate", help="generate output")
    generate_parser.add_argument("-e", "--epoch", type=int, help="specify epoch")
    generate_parser.add_argument("-n", "--nice", type=bool, help="generate nicer output", default=False)
    generate_parser.set_defaults(handler=command_generate)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
