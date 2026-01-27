# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from collections import defaultdict, Counter

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import torch

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    ResourcePoolManager, 
    WorkerType, 
    _timer, 
    # compute_data_metrics, 
    compute_timing_metrics, 
    dataprotoitem_to_dataproto, 
    # compute_advantage, 
    reduce_metrics
)
from verl.utils.torch_functional import masked_mean


# directly copied from verl/trainer/ppo/ray_trainer.py
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, grpo_use_std=True):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo_split':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        prefix_mask = data.batch['prefix_mask']
        on_policy_mask = ~prefix_mask.any(-1)
        from .mix_core_alg import compute_grpo_outcome_advantage_split
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            on_policy_mask=on_policy_mask,
            use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'reinforce':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                             eos_mask=response_mask,
                                                                             index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

class MIXRayPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_entropy_collection()

    def _init_entropy_collection(self):
        """Initialize entropy collection state once."""
        if getattr(self, '_entropy_init_done', False):
            return

        self.collect_entropy = os.getenv('COLLECT_ENTROPY', '0') == '1'
        self._entropy_debug_init_logged = False
        self.sample_counter = 0

        if self.collect_entropy:
            self.entropy_data = {
                'correct_rollouts': [],
                'wrong_rollouts': [],
                'golden_trajectories': [],
            }
            self.entropy_tensors = {
                'correct_rollouts': [],
                'wrong_rollouts': [],
                'golden_trajectories': [],
            }
            # Default to a relative path under the current working directory for open-source use.
            entropy_output_dir = os.getenv('ENTROPY_OUTPUT_DIR', './results/entropy_data')
            os.makedirs(entropy_output_dir, exist_ok=True)
            self.entropy_output_dir = entropy_output_dir
            self.entropy_save_freq = int(os.getenv('ENTROPY_SAVE_FREQ', '10'))
            if not self._entropy_debug_init_logged:
                print(f"[DEBUG][trainer_init] collect_entropy=1, ENTROPY_OUTPUT_DIR={self.entropy_output_dir}, ENTROPY_SAVE_FREQ={self.entropy_save_freq}")
                self._entropy_debug_init_logged = True

        self._entropy_init_done = True

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'grpo_split':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from torch.utils.data import DataLoader, SequentialSampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        from .rl_dataset_with_target import RLHFDatasetWithTarget
        self.train_dataset = RLHFDatasetWithTarget(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True, return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len,
                                         filter_targets=self.config.data.get('filter_targets', False),
                                         sample_target_ratio=self.config.data.get('sample_target_ratio', 1.0))

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            from verl.mix_src.rl_dataset_with_target import ResumableRandomSampler
            sampler = ResumableRandomSampler(data_source=self.train_dataset)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)
        
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        n_samples = self.config.actor_rollout_ref.rollout.n
        if self.config.data.get('add_tgt_with_acc', False):
            n_samples = n_samples - 1 # if filter tgt with acc, we either use tgt or on policy samples.

        for _ in range(self.config.trainer.total_epochs):
            
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                metrics = {}
                timing_raw = {}

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                gen_batch.meta_info['global_steps'] = self.global_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    
                    # This code matches a prompt ID with its N responses.
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # keep golden targets on batch so they survive repeat/union
                    if 'tgt_input_ids' in gen_batch.batch:
                        batch.batch['tgt_input_ids'] = gen_batch.batch['tgt_input_ids']
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # log avg prefix ratio
                    if 'prefix_ratios' in gen_batch_output.meta_info.keys():
                        metrics['batch/avg_prefix_ratio'] = float(np.mean(gen_batch_output.meta_info['prefix_ratios']))
                    
                    if self.config.trainer.add_full_target_when_none:
                        pass

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor = self.reward_fn(batch) # [bsz, l], only the last valid token has reward

                        batch.batch['token_level_scores'] = reward_tensor
                        
                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        
                        if self.config.data.reward_impl_version == 0:
                            fail_value = 0
                            success_value = 1
                            format_value = -1 # not defined.
                        elif self.config.data.reward_impl_version == 1:
                            fail_value = -0.5
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 2:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 3:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 4:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        else:
                            raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')
                        
                        solve_none = 0
                        solve_all = 0
                        solve_none_format = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == fail_value).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == success_value).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                            elif (uid_rewards == format_value).all():
                                valid_mask[uid_mask] = False
                                solve_none_format += 1

                        if self.config.trainer.skip_valid_mask:
                            valid_mask[:] = True
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_none_format'] = solve_none_format
                        metrics['batch/solve_all'] = solve_all

                        # add more metrics
                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)
                        # add on-policy metrics
                        prefix_mask = batch.batch['prefix_mask']
                        off_policy_mask = prefix_mask.any(-1)
                        on_policy_mask = ~off_policy_mask
                        metrics['batch/on_solved'] = (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item() / (on_policy_mask.sum().item() + 1e-6)
                        metrics['batch/off_solved'] = (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item() / (off_policy_mask.sum().item() + 1e-6)
                        
                        # recompute old_log_probs
                        with _timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)
                            if self.collect_entropy:
                                def _has_entropy(container):
                                    try:
                                        return 'entropy' in container
                                    except Exception:
                                        return False

                                def _list_keys(container):
                                    try:
                                        if hasattr(container, 'keys'):
                                            return list(container.keys())
                                        return 'no-keys-attr'
                                    except Exception:
                                        return 'keys-error'

                                if hasattr(old_log_prob, 'batch') and _has_entropy(old_log_prob.batch):
                                    print(f"[DEBUG] Step {self.global_steps}: Entropy found in old_log_prob.batch")
                                if hasattr(batch, 'batch') and _has_entropy(batch.batch):
                                    print(f"[DEBUG] Step {self.global_steps}: Entropy found in merged batch.batch, collecting data...")
                                    self._collect_entropy_data(batch, reward_tensor)
                                else:
                                    available_keys = _list_keys(batch.batch) if hasattr(batch, 'batch') else 'N/A'
                                    print(f"[DEBUG] Step {self.global_steps}: Entropy collection enabled but 'entropy' not in batch.batch. Available keys: {available_keys}")
                                    if hasattr(old_log_prob, 'batch'):
                                        result_keys = _list_keys(old_log_prob.batch)
                                        print(f"[DEBUG] Step {self.global_steps}: old_log_prob.batch keys: {result_keys}")

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # NOTE: the advantages are the same for all tokens in the response
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  grpo_use_std=self.config.algorithm.grpo_use_std)
                            
                        # compute alpha and beta for prefix reward weighting
                        prefix_mask = batch.batch['prefix_mask']
                        advantages = batch.batch['advantages']
                        assert prefix_mask.shape == advantages.shape
                        
                        alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        beta_weight = (~prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        prefix_weight = alpha_weight + beta_weight
                        batch.batch['advantages'] = prefix_weight * advantages
                        
                        if self.config.data.get('disable_truncation_advantage', False):
                            responses = batch.batch['responses']
                            responses_mask = responses != self.tokenizer.pad_token_id
                            response_length = responses_mask.sum(-1) # [bsz]
                            max_len = self.config.data.max_response_length
                            has_truncated = response_length >= max_len
                            no_eos = ~((responses == self.tokenizer.eos_token_id).any(-1))
                            truncated_mask = has_truncated & no_eos
                            batch.batch['advantages'][truncated_mask] = 0

                        if self.config.actor_rollout_ref.actor.get('use_sft_prefix_reward', False):
                            assert self.config.actor_rollout_ref.rollout.n_prefix == -1
                            reward_weight = self.config.actor_rollout_ref.actor.get('sft_prefix_reward_weight', 1.0)
                            batch.batch['advantages'][prefix_mask] = reward_weight / n_samples
                    
                    if self.config.trainer.debug is True:
                        breakpoint()
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        if 'avg_score' not in val_metrics:
                            val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                        metrics.update(val_metrics)
                        if self.config.trainer.get('save_best_ckpt', True):
                            self.maybe_save_best_hf(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        if self.config.trainer.get('save_freq_ckpt', True):
                            with _timer('save_checkpoint', timing_raw):
                                self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics_ours(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.collect_entropy:
                        self._save_entropy_data()
                    return

    def _collect_entropy_data(self, batch: DataProto, reward_tensor: torch.Tensor):
        """Collect entropy data for correct/wrong rollouts and golden trajectories."""
        try:
            has_entropy = 'entropy' in batch.batch
        except Exception:
            has_entropy = False
        if not has_entropy:
            return

        entropy = batch.batch['entropy']
        responses = batch.batch['responses']
        prompts = batch.batch['input_ids']
        attention_mask = batch.batch['attention_mask']
        response_length = responses.shape[1]
        response_mask = attention_mask[:, -response_length:]

        if self.config.data.reward_impl_version in (0, 2, 3, 4):
            fail_value = 0
            success_value = 1
        elif self.config.data.reward_impl_version == 1:
            fail_value = -0.5
            success_value = 1
        else:
            fail_value = 0
            success_value = 1

        sample_rewards = reward_tensor.sum(dim=1)
        correct_mask = (sample_rewards == success_value)
        wrong_mask = (sample_rewards == fail_value)

        batch_size = responses.shape[0]
        for i in range(batch_size):
            prompt_ids = prompts[i]
            response_ids = responses[i]
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)

            per_token_entropy = entropy[i].cpu()
            valid_mask = response_mask[i].cpu().bool()
            per_token_entropy_valid = per_token_entropy[valid_mask]

            category = 'correct_rollouts' if correct_mask[i] else 'wrong_rollouts'
            sample_info = {
                'sample_id': self.sample_counter,
                'step': self.global_steps,
                'prompt': prompt_text,
                'response': response_text,
                'prompt_ids': prompt_ids.cpu().tolist(),
                'response_ids': response_ids.cpu().tolist(),
                'reward': float(sample_rewards[i].item()),
                'entropy_tensor_idx': len(self.entropy_tensors[category]),
                'entropy_tensor_file': None,
            }

            if correct_mask[i]:
                self.entropy_data['correct_rollouts'].append(sample_info)
                self.entropy_tensors['correct_rollouts'].append(per_token_entropy_valid)
            elif wrong_mask[i]:
                self.entropy_data['wrong_rollouts'].append(sample_info)
                self.entropy_tensors['wrong_rollouts'].append(per_token_entropy_valid)

            self.sample_counter += 1

        if 'tgt_input_ids' in batch.batch:
            tgt_input_ids = batch.batch['tgt_input_ids']  # [bsz, max_target_length]
            full_input_ids = batch.batch['input_ids']  # [bsz, prompt_length + generated_response_length]
            generated_responses = batch.batch['responses']  # [bsz, generated_response_length]
            
            # Extract just the prompt portion from full_input_ids
            generated_response_length = generated_responses.shape[1]
            prompt_length = full_input_ids.shape[1] - generated_response_length
            prompts = full_input_ids[:, :prompt_length]  # [bsz, prompt_length]
            
            # Filter out samples with empty golden responses (all padding)
            golden_response_valid = (tgt_input_ids != self.tokenizer.pad_token_id).any(dim=1)  # [bsz]
            
            if golden_response_valid.sum() == 0:
                # No valid golden responses, skip
                pass
            else:
                # Only process samples with valid golden responses
                valid_indices = golden_response_valid.nonzero(as_tuple=True)[0]
                tgt_input_ids_valid = tgt_input_ids[valid_indices]
                prompts_valid = prompts[valid_indices]
                
                # Concatenate prompt with golden response to create the full sequence
                golden_input_ids = torch.cat([prompts_valid, tgt_input_ids_valid], dim=1)  # [valid_bsz, prompt_length + max_target_length]
                
                # Create attention_mask for golden sequence (1 for real tokens, 0 for padding)
                prompt_mask = batch.batch['attention_mask'][valid_indices, :prompt_length]  # [valid_bsz, prompt_length]
                golden_response_mask = (tgt_input_ids_valid != self.tokenizer.pad_token_id).long()  # [valid_bsz, max_target_length]
                golden_attention_mask = torch.cat([prompt_mask, golden_response_mask], dim=1)  # [valid_bsz, prompt_length + max_target_length]
                
                # Create position_ids for golden sequence (must be contiguous)
                golden_seq_len = golden_input_ids.shape[1]
                golden_position_ids = torch.arange(golden_seq_len, device=golden_input_ids.device).unsqueeze(0).repeat(golden_input_ids.shape[0], 1)
                
                # Create a batch with golden trajectories as responses
                golden_batch = DataProto.from_single_dict({
                    'input_ids': golden_input_ids.contiguous(),
                    'responses': tgt_input_ids_valid.contiguous(),
                    'attention_mask': golden_attention_mask.contiguous(),
                    'position_ids': golden_position_ids.contiguous(),
                })
                golden_batch.meta_info = {
                    'micro_batch_size': batch.meta_info.get('micro_batch_size', 32),
                    'max_token_len': batch.meta_info.get('max_token_len', 32768),
                    'use_dynamic_bsz': batch.meta_info.get('use_dynamic_bsz', False),
                    'temperature': batch.meta_info.get('temperature', 1.0),
                }

                try:
                    golden_result = self.actor_rollout_wg.compute_log_prob(golden_batch)
                    if isinstance(golden_result.batch, dict) and 'entropy' in golden_result.batch:
                        golden_entropy = golden_result.batch['entropy']
                        # Note: tgt_input_ids is repeated n times per prompt, so we deduplicate
                        # Map valid_indices back to original indices for deduplication
                        n_samples = self.config.actor_rollout_ref.rollout.n
                        valid_batch_size = tgt_input_ids_valid.shape[0]
                        
                        # Track which original prompts we've already processed
                        processed_prompts = set()
                        for result_idx in range(valid_batch_size):
                            orig_idx = valid_indices[result_idx].item()
                            prompt_idx = orig_idx // n_samples  # Original prompt index
                            
                            if prompt_idx in processed_prompts:
                                continue
                            processed_prompts.add(prompt_idx)
                            
                            prompt_ids = prompts_valid[result_idx]
                            target_ids = tgt_input_ids_valid[result_idx]
                            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                            target_text = self.tokenizer.decode(target_ids, skip_special_tokens=False)

                            per_token_entropy = golden_entropy[result_idx].cpu()
                            valid_mask = (target_ids != self.tokenizer.pad_token_id).cpu().bool()
                            per_token_entropy_valid = per_token_entropy[valid_mask]

                            sample_info = {
                                'sample_id': self.sample_counter,
                                'step': self.global_steps,
                                'prompt': prompt_text,
                                'response': target_text,
                                'prompt_ids': prompt_ids.cpu().tolist(),
                                'response_ids': target_ids.cpu().tolist(),
                                'reward': None,
                                'entropy_tensor_idx': len(self.entropy_tensors['golden_trajectories']),
                                'entropy_tensor_file': None,
                            }

                            self.entropy_data['golden_trajectories'].append(sample_info)
                            self.entropy_tensors['golden_trajectories'].append(per_token_entropy_valid)
                            self.sample_counter += 1
                except Exception as e:
                    import traceback
                    print(f"Warning: Failed to compute golden trajectory entropy: {e}")
                    traceback.print_exc()

        if self.global_steps % self.entropy_save_freq == 0:
            print(f"[DEBUG] Step {self.global_steps}: Triggering entropy save (save_freq={self.entropy_save_freq})")
            self._save_entropy_data()

    def _save_entropy_data(self):
        """Save collected entropy data to disk."""
        total_samples = (
            len(self.entropy_data.get('correct_rollouts', [])) +
            len(self.entropy_data.get('wrong_rollouts', [])) +
            len(self.entropy_data.get('golden_trajectories', []))
        )
        if total_samples == 0:
            print(f"[DEBUG] Step {self.global_steps}: No entropy data to save (all categories empty)")
            return

        step_dir = os.path.join(self.entropy_output_dir, f'step_{self.global_steps}')
        os.makedirs(step_dir, exist_ok=True)
        print(f"[DEBUG] Step {self.global_steps}: Saving entropy data to {step_dir} (total samples: {total_samples})")

        for category in ['correct_rollouts', 'wrong_rollouts', 'golden_trajectories']:
            if len(self.entropy_tensors.get(category, [])) > 0:
                tensor_dir = os.path.join(step_dir, f'{category}_tensors')
                os.makedirs(tensor_dir, exist_ok=True)

                for idx, sample_info in enumerate(self.entropy_data[category]):
                    if sample_info['entropy_tensor_idx'] == idx:
                        tensor_file = os.path.join(tensor_dir, f'sample_{sample_info["sample_id"]}_entropy.npy')
                        np.save(tensor_file, self.entropy_tensors[category][idx].numpy())
                        sample_info['entropy_tensor_file'] = f'{category}_tensors/sample_{sample_info["sample_id"]}_entropy.npy'

        json_data = {
            'step': self.global_steps,
            'correct_rollouts': [
                {k: v for k, v in sample.items() if k not in ['entropy_tensor_idx']}
                for sample in self.entropy_data['correct_rollouts']
            ],
            'wrong_rollouts': [
                {k: v for k, v in sample.items() if k not in ['entropy_tensor_idx']}
                for sample in self.entropy_data['wrong_rollouts']
            ],
            'golden_trajectories': [
                {k: v for k, v in sample.items() if k not in ['entropy_tensor_idx']}
                for sample in self.entropy_data['golden_trajectories']
            ],
            'stats': {
                'correct_rollouts': {'count': len(self.entropy_data['correct_rollouts'])},
                'wrong_rollouts': {'count': len(self.entropy_data['wrong_rollouts'])},
                'golden_trajectories': {'count': len(self.entropy_data['golden_trajectories'])},
            },
        }

        json_file = os.path.join(step_dir, 'samples.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Saved entropy data to {step_dir}")
        print(f"  - JSON: {json_file}")
        print(f"  - Tensors: {step_dir}/*_tensors/")

        cumulative_dir = os.path.join(self.entropy_output_dir, 'latest')
        if os.path.exists(cumulative_dir):
            shutil.rmtree(cumulative_dir)
        shutil.copytree(step_dir, cumulative_dir)
        # Reset buffers so each step dir only contains fresh samples
        for k in self.entropy_data.keys():
            self.entropy_data[k] = []
        for k in self.entropy_tensors.keys():
            self.entropy_tensors[k] = []

    def maybe_save_best_hf(self, val_metrics: dict):
        import json
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'best',
                                        f'actor')
        
        os.makedirs(actor_local_path, exist_ok=True)
        if os.path.exists(f'{actor_local_path}/metrics.json'):
            with open(f'{actor_local_path}/metrics.json', 'r') as f:
                metrics = json.load(f)
            best_score = metrics['best_avg_score']
        else:
            print('Find no current best saved. Best score is set to -inf')
            best_score = -float('inf')
        
        cur_score = val_metrics['avg_score']
        
        if cur_score > best_score:
            print(f'Saving best checkpoint with score {cur_score} at {actor_local_path}')
            best_score = cur_score
            self.actor_rollout_wg.save_checkpoint_hf(actor_local_path)
            with open(f'{actor_local_path}/metrics.json', 'w') as f:
                f.write(json.dumps({'best_avg_score': best_score, 'global_step': self.global_steps})+'\n')
        
def compute_data_metrics_ours(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    from verl.trainer.ppo.ray_trainer import _compute_response_info
    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # compute on/off policy stats
    off_policy_mask = batch.batch['prefix_mask'].any(-1) # [bsz, ]
    on_policy_mask = ~off_policy_mask
    off_response_length = response_length[off_policy_mask]
    on_response_length = response_length[on_policy_mask]
    
    off_on_example_ratio = off_policy_mask.sum().item() / on_policy_mask.sum().item()

    off_sequence_score = sequence_score[off_policy_mask]
    on_sequence_score = sequence_score[on_policy_mask]

    # on/off prompt score
    # batch_size = batch.batch.batch_size[0] / n_samples
    # on_prompt_score, off_prompt_score = [], []
    # sequence_score = sequence_score.reshape(batch_size, n_samples, sequence_score.shape[-1]) # [bsz, n, l]
    # for i in range(batch_size):
    #     on_prompt_score.append(sequence_score[i][on_policy_mask[i]].mean())
    #     off_prompt_score.append(sequence_score[i][off_policy_mask[i]].mean())

    # on_prompt_score = torch.cat(on_prompt_score, dim=0)
    # off_prompt_score = torch.cat(off_prompt_score, dim=0)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # on/off policy response length
        'on_off_metrics/on_response_length_mean':
            torch.mean(on_response_length).detach().item(),
        'on_off_metrics/off_response_length_mean':
            torch.mean(off_response_length).detach().item(),
        'on_off_metrics/on_score':
            torch.mean(on_sequence_score).detach().item(),
        'on_off_metrics/off_score':
            torch.mean(off_sequence_score).detach().item(),
        # 'on_off_metrics/on_prompt_score':
        #     torch.mean(on_prompt_score).detach().item(),
        # 'on_off_metrics/off_prompt_score':
        #     torch.mean(off_prompt_score).detach().item(),
        'on_off_metrics/off_on_example_ratio':
            off_on_example_ratio,
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics