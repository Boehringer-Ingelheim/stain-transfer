import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from src import MODELS, TISSUES, NAMELESS_MODEL
from src.data_processing.datasets import get_img_files

# Regex pattern
MT_PATTERN = re.compile("|".join(f"_{m}" for m in MODELS + TISSUES))


def mt_rename(some_string):
    return re.sub(MT_PATTERN, "", some_string)


def group_files(f: list) -> defaultdict:
    replaced = [mt_rename(file.name) for file in f]
    total_models = replaced.count(max(replaced, key=replaced.count))
    grouped_files = defaultdict(list)
    if total_models > 1:
        for file in f:
            if replaced.count(mt_rename(file.name)) == total_models:
                located = False
                for m in MODELS:
                    if m in file.name:
                        grouped_files[m].append(file)
                        located = True
                        break
                if not located:
                    grouped_files['Real'].append(file)

    return grouped_files


def get_triplets(source_path: str, fake_path: str, target_path: str = None):
    """
    Gets a dictionary of triplets for paired datasets: source, fake, target.
    The dictionary maps the names of source images to actual file paths, for
    each of the detected models and for the target path.

    :param source_path: path to real source images
    :param fake_path: path to fake generated images
    :param target_path: path to real target images
    :return: Triplet dictionary
    """

    triplet = {}
    source, fake = Path(source_path), Path(fake_path)
    keys, paths = ["source", "fake"], [source, fake]
    if target_path is not None:
        target = Path(target_path)
        keys.append("target")
        paths.append(target)
    else:
        triplet['target'] = {}

    for t, p in zip(keys, paths):
        if not p.exists():
            raise RuntimeError(f"Invalid {t} path: {p}.")
        if t != 'fake':
            files = get_img_files(p)
            triplet[t] = {mt_rename(Path(x).stem): x for x in files}

    files = get_img_files(fake)
    triplet['fakes'] = {}
    for m in MODELS + [NAMELESS_MODEL]:
        m_files = [x for x in files if m in Path(x).name]
        # Mandatory since everything matches NAMELESS_MODEL.
        if m == NAMELESS_MODEL:
            # In case images were generated without suffixes, then infer model
            # from fakes path.
            model = intersect_lists(fake.parts, MODELS)
            if len(model):
                m = model[0]
        if m != NAMELESS_MODEL:
            triplet['fakes'][m] = {mt_rename(Path(x).stem): x for x in m_files}

    return triplet


def path2fn(path: str):
    """
    Converts from a path to a filename by replacing file system separators,
    spaces and forbidden characters with underscores.

    :param path: Path to be converted
    :return: Converted filename without empty spaces or separators.
    """

    return path.replace(' ', '_').replace('/', '_').replace('\0', '_').replace(
        '\\', '_').replace(':', '_')


def intersect_lists(*args: list) -> list:
    """
    Returns common elements in lists.

    :param args: lists to be intersected.
    :return: list containing the common elements across all passed lists.
    """

    for i, x in enumerate(args):
        if i == 0:
            matches = set(x)
        else:
            matches = matches.intersection(set(x))

    return list(matches)


def multicolumn2csv(df: pd.DataFrame, csv_fn: str) -> None:
    """
    Renames duplicated column names to '' on the top level when saving a dataframe to a csv.

    :param df: DataFrame to be renamed.
    :param csv_fn: file name.
    """

    duplicated = df.columns.droplevel(level=1).duplicated()
    columns = []
    for i, x in enumerate(duplicated):
        if x:
            columns.append(('', df.columns.values[i][1]))
        else:
            columns.append(df.columns.values[i])
    df.columns = pd.MultiIndex.from_tuples(columns)
    df.to_csv(csv_fn)


def csv2multicolumn(csv_fn: str) -> pd.DataFrame:
    """
    Reads a csv file into a multicolumn DataFrame renaming empty column names.
    Must be a csv saved with <multicolumn2csv>.

    :param csv_fn: file name.
    :return: DataFrame with corrected column names.
    """

    df = pd.read_csv(csv_fn, index_col=0, header=[0, 1])
    columns = []
    last_name = ""
    for i, x in enumerate(df.columns.values):
        if "Unnamed: " in x[0]:
            columns.append((last_name, df.columns.values[i][1]))
        else:
            last_name = df.columns.values[i][0]
            columns.append(df.columns.values[i])
    df.columns = pd.MultiIndex.from_tuples(columns)

    return df


def check_exists(path: str):
    """
    Checks if a path/file exists.

    :param path: Path/file to be checked.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} doesn't exist")


def check_path(path: str):
    """
    Checks if a path exists.

    :param path: Path to be checked.
    """
    if not Path(path).is_dir():
        raise NotADirectoryError(f"Path {path} doesn't exist")


def get_config(conf: str):
    """
    Gets configuration from a yaml file.

    :param conf: path to yaml file.
    :return: configuration dictionary.
    """

    with open(conf, "r") as config:
        try:
            conf = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            raise exc
    return conf
