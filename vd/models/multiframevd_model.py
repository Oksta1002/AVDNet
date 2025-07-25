import torch
import os.path as osp
import time
from tqdm import tqdm
from collections import OrderedDict
from vd.utils.registry import MODEL_REGISTRY
from vd.archs import build_network
from vd.models import BaseModel
from vd.metrics import calculate_metric
from vd.utils import get_root_logger
from vd.losses import build_loss
from vd.data.data_util import tensor2numpy, imwrite_gt
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


@MODEL_REGISTRY.register()
class MultiFrameVDModel(BaseModel):
    def __init__(self, opt):
        super(MultiFrameVDModel, self).__init__(opt)

        # define networks
        self.net = build_network(opt['network'])
        self.net = self.model_to_device(self.net)
        with get_accelerator().device(0):
            flops, macs, params = get_model_profile(model=self.net, # model
                                    input_shape=(1, 3, 3, 720, 1280), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling
            logger = get_root_logger()
            logger.info(f'flops: {flops}, macs: {macs}, params: {params}')

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key', 'params')
            self.load_network(self.net, load_path, self.opt['path'].get('strict_load', True), param_key)
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net with Exponential Moving Average (EMA)
            # net_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_ema = build_network(self.opt['network']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network', None)
            if load_path is not None:
                self.load_network(self.net_ema, load_path, self.opt['path'].get('strict_load', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net weight
            self.net_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'gts' in data:
            self.gts = data['gts'].to(self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim'].pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **train_opt['optim'])
        self.optimizers.append(self.optimizer)

    def optimize_parameters(self, current_iter):
        self.optimizer.zero_grad()
        self.output = self.net(self.lq)
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        l_total.backward()
        self.optimizer.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        scale = self.opt.get('scale', 1)
        _, _, _, h_old, w_old = self.lq.size()
        if hasattr(self, 'net_ema'):
            self.net_ema.eval()
            with torch.no_grad():
                self.output = self.net_ema(self.lq)
                self.output = self.output[:, :, :h_old * scale, :w_old * scale]
        else:
            self.net.eval()
            with torch.no_grad():
                self.output = self.net(self.lq)
                self.output = self.output[:, :, :h_old * scale, :w_old * scale]
            self.net.train()
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        time_inf_total = 0.
        for idx, val_data in enumerate(dataloader):
            img_name = val_data['key'][0]
            self.feed_data(val_data)
            st = time.time()
            self.test()
            st1 = time.time() - st
            time_inf_total += st1
            visuals = self.get_current_visuals()
            if self.opt['is_train']:
                sr_img_tensors = self.output.detach()
                metric_data['img'] = sr_img_tensors
                if 'gt' in visuals:
                    gt_img_tensors = self.gt.detach()
                    metric_data['img2'] = gt_img_tensors
                    del self.gt
            else:
                sr_img = tensor2numpy(visuals['result'])
                metric_data['img'] = sr_img
                if 'gt' in visuals:
                    gt_img = tensor2numpy(visuals['gt'])
                    metric_data['img2'] = gt_img
                    del self.gt
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            if save_img:
                if self.opt['is_train']:
                    pass
                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                imwrite_gt(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    if self.opt['is_train']:
                        self.metric_results[name] += calculate_metric(metric_data, opt_).detach().cpu().numpy().sum()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()
        time_avg = time_inf_total / (idx + 1)
        logger = get_root_logger()
        logger.info('average test time: %.3f, total time: %.3f' % (time_avg, time_inf_total))
        if with_metrics:
            for metric in self.metric_results.keys():
                if self.opt['is_train']:
                    self.metric_results[metric] /= 2580
                else:
                    self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_ema'):
            self.save_network([self.net, self.net_ema], 'net', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net, 'net', current_iter)
        self.save_training_state(epoch, current_iter)
