import importlib
from os import path as osp
from copy import deepcopy
from vd.utils.registry import METRIC_REGISTRY
from vd.utils import scandir
from .psnr import *
from .ssim import *

# automatically scan and import metric modules for registry
# scan all the files that end with 'metric.py' under the metrics folder
metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(metric_folder) if v.endswith('_metric.py')]
# import all the metric modules
_metric_modules = [importlib.import_module(f'vd.metrics.{file_name}') for file_name in metric_filenames]

def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric