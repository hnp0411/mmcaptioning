import random

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer)
from mmcv.utils import build_from_cfg

from mmcap.core import (TextGenerateRunner, 
                     DistEvalHook, EvalHook)
from mmcap.datasets import (build_dataloader, build_dataset,
                         replace_ImageToTensor)
from mmcap.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_caption_model(model,
                        dataset,
                        cfg,
                        distributed=False,
                        validate=False,
                        timestamp=None,
                        meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        #find_unused_parameters = cfg.get('find_unused_parameters', False)
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print("N_PARAMETERS", n_parameters)
    print('--------------------------------------')
    # build runner
    # AdamW Optimizer
    # TODO -> build_optimizer 구현

    param_dicts = [
        {"names": [n for n, p in model.named_parameters() \
                if "backbone" in n and p.requires_grad],
         "params": [p for n, p in model.named_parameters() \
                if "backbone" in n and p.requires_grad],
         "lr": cfg.lr_dict.lr_backbone},
        {"names": [n for n, p in model.named_parameters() \
                if "backbone" not in n and p.requires_grad],
         "params": [p for n, p in model.named_parameters() \
                if "backbone" not in n and p.requires_grad]},
    ]
    #optimizer = build_optimizer(model, cfg.optimizer)
    optimizer = torch.optim.AdamW(
        param_dicts, lr=cfg.lr_dict.lr, weight_decay=cfg.weight_decay)

    # nondistubuted -> TextGenerateRunner
    # distributed -> EpochBasedRunner
    if not distributed:
        runner = TextGenerateRunner(
            model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta)
        # default 50 batch 마다 하나의 샘플에 대해서 문장 생성함
        runner.set_gen_iter(cfg.log_config.interval) 
        # set tokenizer for train sample generation
        runner.set_tokenizer(dataset[0].tokenizer) 
        # set decoding method for train sample generation
        runner.set_decoding_cfg(cfg.train_cfg.decoding_cfg)
    else: # distributed
        runner = EpochBasedRunner(
            model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta)

    # an ugly workaround to make .log and .log.json filenames the same
    # TODO -> Docker 시간 설정 
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # TODO : Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
