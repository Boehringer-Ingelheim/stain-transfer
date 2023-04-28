import argparse
import traceback

from src.data_processing.config import GenerateConf, MetricsConf
from src.data_processing.metrics import generate_classic_metrics
from src.modelling.rotated import rotate_model
from src.utils.utils import get_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate fake images and/or calculate metrics')
    parser.add_argument('--conf', required=True, help='Configuration yaml.')
    parser.add_argument('--generate', action='store_true',
                        help='if specified, generate fake images.')
    parser.add_argument('--metrics', action='store_true',
                        help='if specified, generate metrics.')
    args = parser.parse_args()

    try:
        conf = get_config(args.conf)
        if args.generate:
            generate_conf = GenerateConf(**conf['generate'])
            print(generate_conf)
            for i, model in enumerate(generate_conf.models.models):
                generate_conf.weights = generate_conf.models.weights[i]
                if generate_conf.rotate:
                    model = rotate_model(model)
                model = model(generate_conf)
                rotated = model.predict()
                if rotated is not None:
                    print('\n'.join(
                        {f"{k}°: SSIM {rotated[k]}" for k in sorted(rotated)}))

        if args.metrics:
            metrics = generate_classic_metrics(MetricsConf(**conf['metrics']))
            print(metrics.mean())
    except Exception as e:
        traceback.print_exc()
        print(f"ERROR: {e}")
