import argparse
import logging
import os
import datetime

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import set_random_seed, Runner

from mmseg.registry import RUNNERS
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('model_name', type=str, 
        help='model name, used for creating experiment folder')
    parser.add_argument('--work-dir', type=str, default='./output',
        help='working folder which contains checkpoint files, log, etc.')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--wandb', default=False, action=argparse.BooleanOptionalAction, help='Use wandb')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args_to_config(cfg, args):
    # Set up working dir to save files and logs.
    run_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = f'/{args.model_name}/exp_{run_time}'
    cfg.work_dir =  args.work_dir + exp_name

    # Set seed thus the results are more reproducible
    # cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # support wandb
    cfg.visualizer.vis_backends.append(dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='transformer-material-segmentation',
            name=f'{exp_name}',
            group=args.model_name)
        )
        )
    
    cfg.launcher=args.launcher

    return cfg



def main():
    args = parse_args()

    if args.wandb:
        wandb.login(key='fe0119224af6709c85541483adf824cec731879e')

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    cfg = merge_args_to_config(cfg, args)

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'


    # resume training
    cfg.resume = args.resume
    
    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
