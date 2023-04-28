from src.data_processing.metric_inspection import inspect

# Use this visualizer to inspect images and their associated metrics.
# Modify the configuration dictionary to change experiments.

# source      (str): path to input real source images.
# target      (str): path to expected real target images.
# fake        (str): path to generated fake images.
# stats_file  (str): file containing the statistics for the specified source,
#                    target and fake paths. Must have been generated with the
#                    --metrics option from the main.py file.
# stats (list[str]): list of statistics to inspect. Available ones are:
#                    ["ssim", "fsim", "wasserstein", "hellinger",
#                    "kullback_leibler", "bhattacharyya"]

source = 'data/paired/images_a'
target = 'data/paired/images_b'
fake = 'results/images/all_fakes/paired_he2mt'
conf = {"source": source,
        "target": target,
        "fake": fake,
        "stats_file": "results/statistics/paired_H_E_all_fakes.csv",
        "stats": ["ssim", "fsim", "wasserstein"]}

if __name__ == '__main__':
    inspect(**conf)
