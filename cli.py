from argparse import ArgumentParser


def getParser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["linear", "xor"],
        default="xor",
        help="choose dataset to train [linear, xor]",
    )
    parser.add_argument(
        "-s",
        "--sample_size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=890104,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["momentum", "gd"],
        default="momentum",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "bce"],
        default="bce",
    )
    return parser
