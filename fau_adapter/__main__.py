"""The command line: fau-adapter {targets,train,generate} (or python -m fau_adapter)."""
import argparse

from . import __version__, generate, targets, train


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="fau-adapter",
        description="Adapt a released distilled text-to-image model so that its images train better classifiers. "
                    "Three steps: `targets` runs the unguided base model to make training pairs, `train` fits the "
                    "adapter, `generate` makes images with it. Each command has its own --help.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="{targets,train,generate}")
    targets.add_parser(sub)
    train.add_parser(sub)
    generate.add_parser(sub)
    args = ap.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
