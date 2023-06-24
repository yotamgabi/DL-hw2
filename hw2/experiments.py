import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from cs3600.train_results import FitResult

from . import cnn, training

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

MODEL_TYPES = dict(
    cnn=cnn.ConvClassifier, resnet=cnn.ResNetClassifier, ycn=cnn.YourCodeNet
)


def run_experiment(
        run_name,
        out_dir="./results",
        seed=None,
        device=None,
        # Training params
        bs_train=128,
        bs_test=None,
        batches=100,
        epochs=100,
        early_stopping=3,
        checkpoints=None,
        lr=1e-3,
        reg=1e-3,
        # Model params
        filters_per_layer=None,
        layers_per_block=None,
        pool_every=None,
        hidden_dims=None,
        model_type="cnn",
        # Additional parameters to consider:
        # in_channels=3,
        # conv_params=dict(kernel_size=3, stride=1, padding=1),
        pooling_params=dict(kernel_size=2),

        # You can add extra configuration for your experiments here
        **kw,
):
    """
    Executes a single run of a Part3 experiment with a single configuration.

    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)

    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======
    # Data - use DataLoader
    x0, _ = ds_train[0]
    in_size = x0.shape
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=bs_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=bs_test)
    """
    resnet:

                in_size,
                out_classes,
                channels,
                pool_every,
                hidden_dims,
                batchnorm=False,
                dropout=0.0,
                **kwargs,
    conv:

                in_size,
                out_classes: int,
                channels: Sequence[int],
                pool_every: int,
                hidden_dims: Sequence[int],
                conv_params: dict = {},
                activation_type: str = "relu",
                activation_params: dict = {},
                pooling_type: str = "max",
                pooling_params: dict = {},
    """
    # Create model, loss, and optimizer instances
    model = model_cls(
        hidden_dims=hidden_dims,
        in_size=in_size,
        out_classes=10,
        channels=filters_per_layer * layers_per_block,
        pool_every=pool_every,
        pooling_params=pooling_params
    ).to(device)

    # Create model, loss and optimizer instances

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    criterion = torch.nn.CrossEntropyLoss()
    # Create a Trainer instance
    trainer = training.TorchTrainer(model, criterion, optimizer, device)

    # Train the model and save the fit results
    fit_res = trainer.fit(
        dl_train=train_loader,
        dl_test=test_loader,
        num_epochs=epochs,
        early_stopping=early_stopping,
    )
    # ========================

    save_experiment(run_name, out_dir, cfg, fit_res)


def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(config=cfg, results=fit_res._asdict())

    cfg_LK = (
        f'L{cfg["layers_per_block"]}_K'
        f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"* Output file {output_filename} written")


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="CS3600 HW2 Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=run_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # # Model
    sp_exp.add_argument(
        "--filters-per-layer",
        "-K",
        type=int,
        nargs="+",
        help="Number of filters per conv layer in a block",
        metavar="K",
        required=True,
    )
    sp_exp.add_argument(
        "--layers-per-block",
        "-L",
        type=int,
        metavar="L",
        help="Number of layers in each block",
        required=True,
    )
    sp_exp.add_argument(
        "--pool-every",
        "-P",
        type=int,
        metavar="P",
        help="Pool after this number of conv layers",
        required=True,
    )
    sp_exp.add_argument(
        "--hidden-dims",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=True,
    )
    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if _name_ == "_main_":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"* Starting {subcmd_fn._name_} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))