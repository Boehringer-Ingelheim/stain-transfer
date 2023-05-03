import argparse
from pathlib import Path

import pandas as pd

from src.data_processing.config import MetricsConf
from src.data_processing.metrics import (generate_classic_metrics,
                                         unpaired_lab_WD, calculate_fid)

COLS = ['Fake', 'SSIM', 'SSIM_std', 'FID', 'LAB_WD', 'Real Source',
        'Real target']


def update_df(metric_df: pd.DataFrame, fake: str, source: str, target: str,
              ssim: float, ssim_std: float, fid: float, lab_wd: float, output: str):
    df = pd.DataFrame(
        data=[[fake, ssim, ssim_std, fid, lab_wd, source, target]],
        columns=COLS)
    metric_df = pd.concat([metric_df, df])
    metric_df.to_csv(output)

    return metric_df


def format_paths(source: str, target: str, fakes: list, output: str):
    f = '\n\t'.join(['\t'] + [str(x) for x in fakes])
    return f"Source path:\n\t{source}\nTarget path:\n\t{target}\nFake paths:{f}\nSaving output to:\n\t{output}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute SSIM, WD and FID for generated fakes')
    parser.add_argument('--real_source', required=True,
                        help='Path to real source images from which fakes were '
                             'generated. Used for SSIM')
    parser.add_argument('--real_target', required=True,
                        help='Path to real target images. Needed for FID and WD.')
    parser.add_argument('--fakes', required=True,
                        help='Path to folder containing folders of fake images.')
    parser.add_argument('--output', default=None,
                        help='Path to save computed metrics. If not specified '
                             'will be saved in fakes path.')
    parser.add_argument('--device', type=int, default=0,
                        help='Device for running FID computations. -1 for CPU.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for FID')
    parser.add_argument('--crop_size', default=None, type=int,
                        help='Center crop images.')
    args = parser.parse_args()

    device = 'cpu' if args.device < 0 else f"cuda:{args.device}"
    fakes = sorted(Path(args.fakes).glob('*'))
    fakes = [f for f in fakes if f.is_dir()]
    out = fakes[0].parent if args.output is None else args.output
    out = Path(out) / 'metrics.csv'
    paths = format_paths(args.real_source, args.real_target, fakes, out)
    print(paths)
    df = pd.DataFrame(columns=COLS)
    for fake in fakes:
        m = MetricsConf(classic_metrics=['ssim'], center_crop=args.crop_size,
                        source=args.real_source, fake=fake)
        results = generate_classic_metrics(metrics_conf=m)
        ssim_score = results.mean().values[0]
        ssim_std = results.std().values[0]
        lab_wd = unpaired_lab_WD(args.real_target, str(fake))
        fid_score = calculate_fid([args.real_target, str(fake)], device=device,
                                  batch_size=args.batch_size)
        df = update_df(df, fake, args.real_source, args.real_target, ssim_score,
                       ssim_std, fid_score, lab_wd, out)
