"""
Luke
"""
import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad

from mmcv.runner.dist_utils import allreduce_grads
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks import OptimizerHook


@HOOKS.register_module()
class CaptionFp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook.

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float): Scale factor multiplied with loss.
    """

    def __init__(self,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 loss_scale=512.,
                 distributed=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.loss_scale = loss_scale
        self.distributed = distributed

    def before_run(self, runner):
        """Preparing steps before Mixed Precision Training.

        1. Make a master copy of fp32 weights for optimization.
        2. Convert the main model from fp32 to fp16.
        """
        # keep a copy of fp32 weights
        old_groups = runner.optimizer.param_groups
        runner.optimizer.param_groups = copy.deepcopy(
            runner.optimizer.param_groups)
        state = defaultdict(dict)
        p_map = {
            old_p: p
            for old_p, p in zip(
                chain(*(g['params'] for g in old_groups)),
                chain(*(g['params'] for g in runner.optimizer.param_groups)))
        }
        for k, v in runner.optimizer.state.items():
            state[p_map[k]] = v
        runner.optimizer.state = state
        # convert model to fp16
        wrap_fp16_model(runner.model)

    def copy_grads_to_fp32(self, fp16_net, fp32_names, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        fp16_weight_dict = {n:p for n, p in fp16_net.named_parameters() \
                            if p.requires_grad}

        assert(len(fp16_weight_dict) == len(fp32_weights))
        for fp32_name, fp32_param in zip(fp32_names, fp32_weights):
            fp16_param = fp16_weight_dict[fp32_name]
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights_dict):
        require_grads = [p.requires_grad for n, p in fp16_net.named_parameters()]
        fp16_names = [n for n, p in fp16_net.named_parameters()]

        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp16_names, require_grad in \
                zip(fp16_net.parameters(), fp16_names, require_grads):
            if require_grad:
                fp32_param = fp32_weights_dict[fp16_names]
                fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner):
        """Backward optimization steps for Mixed Precision Training.

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients (fp16).
        3. Copy gradients from the model to the fp32 weight copy.
        4. Scale the gradients back and update the fp32 weight copy.
        5. Copy back the params from fp32 weight copy to the fp16 model.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        # scale the loss value
        scaled_loss = runner.outputs['loss'] * self.loss_scale
        scaled_loss.backward()

        # copy fp16 grads in the model to fp32 params in the optimizer
        fp32_weights = []
        fp32_names = []
        for param_group in runner.optimizer.param_groups:
            fp32_weights += param_group['params']
            fp32_names += param_group['names']

        self.copy_grads_to_fp32(runner.model, fp32_names, fp32_weights)
        # allreduce grads
        if self.distributed:
            allreduce_grads(fp32_weights, self.coalesce, self.bucket_size_mb)
        # scale the gradients back
        for param in fp32_weights:
            if param.grad is not None:
                param.grad.div_(self.loss_scale)
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(fp32_weights)
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        # update fp32 params
        runner.optimizer.step()
        # copy fp32 params to the fp16 model
        fp32_weights_dict = {k:v for k,v in zip(fp32_names, fp32_weights)}
        self.copy_params_to_fp16(runner.model, fp32_weights_dict)
