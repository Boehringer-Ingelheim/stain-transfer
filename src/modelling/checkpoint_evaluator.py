import argparse
import re
from pathlib import Path
from shutil import rmtree

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from src import MODELS
from src.data_processing.config import Models, GenerateConf, ModelConf
from src.data_processing.datasets import ImagePathDataset
from src.data_processing.metrics import calculate_fid, unpaired_lab_WD_and_WD, \
    generate_classic_metrics
from src.utils.utils import check_path

LATEST = int(1E20)  # Big int number for latest.
IMG = 'IMAGES'
PAIRED = 'PAIRED'


def get_checkpoint(checkpoint_name: str) -> int:
    """
    Get the chekpoint number from a checkpoint file name. First matching number
    is considered as checkpoint.
    Supports 'latest' and 'last' naming conventions.
    Ex. 5_net_G_A, 25_net_G_B, latest_net_G_A, etc.

    :param checkpoint_name: Checkpoint name
    :return: Integer checkpoint.
    """

    cp_id = re.search(r"[0-9]+", checkpoint_name)
    if cp_id == None:
        if 'latest' or 'last' in checkpoint_name:
            cp_id = LATEST
        else:
            # Other possible name. To be discovered. Setting to LATEST always now.
            cp_id = LATEST
    else:
        cp_id = int(cp_id.group())

    return cp_id


def load_images(image_path: str, batch_size: int,
                num_workers: int) -> torch.Tensor:
    """
    Load images from a directory and make a grid.

    :param image_path: Path to images.
    :param batch_size: Amount of images to load from path.
    :param num_workers: Dataloader workers.
    :return:
    """

    dataset = ImagePathDataset(image_path, transforms.ToTensor())
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    images, paths = next(iter(dataloader))
    grid = make_grid(images)

    return grid


def log_metric_or_image(board: SummaryWriter, key: str, s_name: str,
                        i_name: str, value: float, step: int) -> None:
    """ Log a metric or an image to a board."""

    if key == IMG:
        board.add_image(i_name, value, step)
    else:
        board.add_scalar(s_name, value, step)


def log_results(results: dict, title: dict, board: SummaryWriter) -> None:
    """
    Log results to tensorboard.

    :param results: Nested results dictionary, having as main keys fake generation direction.
    :param board: Tensorboard to write results to.
    """

    for direction, result in results.items():
        for checkpoint in sorted(result):
            step = checkpoint
            if checkpoint == LATEST:
                step = last_checkpoint + 1
            for k, v in result[checkpoint].items():
                if k == PAIRED:
                    for paired_k, paired_v in result[checkpoint][k].items():
                        m_name = f"{k} {title[direction]}/ {paired_k}"
                        i_name = f"{k} {title[direction]}/ Images"
                        log_metric_or_image(board, paired_k, m_name, i_name,
                                            paired_v, step)
                else:
                    m_name = f"NON {PAIRED} {title[direction]}/ {k}"
                    i_name = f"NON {PAIRED} {title[direction]}/ Images"
                    log_metric_or_image(board, k, m_name, i_name, v, step)
            last_checkpoint = checkpoint
    board.flush()


def check_inputs(opt: argparse.Namespace) -> None:
    """Check if paths exist."""

    check_path(opt.checkpoints)
    check_path(opt.domain_A)
    check_path(opt.domain_B)
    check_path(opt.results_path)
    if opt.paired_domain_A is not None and opt.paired_domain_B is not None:
        check_path(opt.paired_domain_A)
        check_path(opt.paired_domain_B)


def set_conf(opt: argparse.Namespace) -> dict:
    """Gets configuration dict from arguments"""

    conf = {"batch_size": opt.batch_size, "results_path": opt.results_path,
            "device": opt.device, "num_workers": opt.num_workers}

    return conf


def generate_fakes(model: str, data_path: str, conf: dict, weights: str) -> str:
    """
    Generate fake images. Return path to generated fake images to run evaluator
    on them later.
    """

    name = [x for x in MODELS if x.lower() in model.name.lower()][0]
    models = Models(names=[name], weights=[weights])
    conf['data_path'] = data_path
    conf = GenerateConf(**conf, models=models)
    print(conf)
    del conf.models
    model_conf = ModelConf(**conf.__dict__)
    model_conf.weights = weights
    Model = models.models[0](model_conf)
    Model.predict()
    fakes = str(Model.image_outs)
    del Model

    return fakes


def main(args):
    """
    Generate fakes and compute metrics in both directions.
    """

    fid_dev = args.device if args.device >= 0 else 'cpu'
    check_inputs(args)
    conf = set_conf(args)
    DIRECTIONS = ['A2B', 'B2A']
    title = {'A2B': f"{Path(args.domain_A).name} to {Path(args.domain_B).name}"}
    title['B2A'] = f"{Path(args.domain_B).name} to {Path(args.domain_A).name}"

    for model in Path(args.checkpoints).glob('*'):
        tb_writer = SummaryWriter(Path(args.results_path) / f"run_{model.name}")
        results = {x: {} for x in DIRECTIONS}
        for c_point in sorted(model.glob('*')):
            for i, direction in enumerate(DIRECTIONS):
                if c_point.stem.endswith(f"G_{direction[0]}"):
                    try:
                        data = args.domain_A if i == 0 else args.domain_B
                        target = args.domain_B if i == 0 else args.domain_A
                        fakes_path = generate_fakes(model, data, conf, c_point)
                        fid = calculate_fid([fakes_path, target],
                                            batch_size=args.batch_size,
                                            device=fid_dev)
                        lab_wd, wd = unpaired_lab_WD_and_WD(fakes_path, target)
                        cp_id = get_checkpoint(c_point.stem)
                        fakes = load_images(fakes_path, args.grid_size,
                                            args.num_workers)
                        results[direction][cp_id] = {'FID': fid,
                                                     'WD': wd,
                                                     'LAB_WD': lab_wd,
                                                     IMG: fakes}
                        rmtree(fakes_path, ignore_errors=True)

                        if args.paired_domain_A is not None and args.paired_domain_B is not None:
                            data = args.paired_domain_A if i == 0 else args.paired_domain_B
                            target = args.paired_domain_B if i == 0 else args.paired_domain_A
                            paired_fakes_path = generate_fakes(model, data,
                                                               conf, c_point)
                            metrics = generate_classic_metrics(data,
                                                               paired_fakes_path,
                                                               target,
                                                               args.results_path)
                            metrics.columns = metrics.columns.droplevel()
                            paired_fakes = load_images(paired_fakes_path,
                                                       args.grid_size,
                                                       args.num_workers)
                            rmtree(paired_fakes_path, ignore_errors=True)
                            results[direction][cp_id][
                                PAIRED] = metrics.mean().to_dict()
                            results[direction][cp_id][PAIRED][
                                IMG] = paired_fakes

                    except Exception as e:
                        log_results(results, title, tb_writer)
                        tb_writer.close()
                        raise e

        # Logging is done at the end because checkpoint file names are not sorted.
        log_results(results, title, tb_writer)
        tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate different checkpoints for cyclegan based generators.'
                    'Checkpoint file stems must end with G_x, where x is either '
                    'A or B; and contain a number indicating the epoch or the '
                    'words last ot latest, Ex. 20_net_G_A.pth, last_net_G_B.pth')
    parser.add_argument('--checkpoints', type=str, required=True,
                        help="Path to directory containing folders of checkpoints."
                             "Each folder of checkpoints should contain the model's name")
    parser.add_argument('--domain_A', type=str, required=True,
                        help='Path to domain A images.')
    parser.add_argument('--domain_B', type=str, required=True,
                        help='Path to domain B images.')
    parser.add_argument('--paired_domain_A', type=str, default=None,
                        help='Path to domain A paired images.')
    parser.add_argument('--paired_domain_B', type=str, default=None,
                        help='Path to domain B paired images.')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to save results to.')
    parser.add_argument('--device', type=int, default=-1,
                        help='Device to run evaluator on.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Grid size for tensorboard.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Dataloader workers.')
    args = parser.parse_args()

    main(args)
