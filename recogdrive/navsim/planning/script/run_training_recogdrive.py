from typing import Tuple
from pathlib import Path
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"




def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    #features_list, targets_list, tokens_list = zip(*batch)

    if len(batch[0]) == 3:
        features_list, targets_list, tokens_list = zip(*batch)
    elif len(batch[0]) == 2:
        features_list, targets_list = zip(*batch)
        tokens_list = tuple([None] * len(batch))
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}. Expected 2 or 3.")

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    # last_hidden_state = rnn_utils.pad_sequence(
    #     [features['last_hidden_state'] for features in features_list],
    #     batch_first=True,
    #     padding_value=0.0
    # ).clone().detach()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        #'last_hidden_state': last_hidden_state,
        'status_feature': status_feature
    }
    
    # Backward-compatible dual path:
    # 1) Cached hidden-state training: batch contains `last_hidden_state`.
    # 2) Online backbone fine-tuning: batch contains `image_path_tensor`.
    if all('last_hidden_state' in f for f in features_list):
        last_hidden_state = rnn_utils.pad_sequence(
            [features['last_hidden_state'] for features in features_list],
            batch_first=True,
            padding_value=0.0
        ).clone().detach()
        features['last_hidden_state'] = last_hidden_state

    if all('pixel_values' in f for f in features_list):
        # Keep as list-like packed tensor [B, P, C, H, W] with per-sample patch count P potentially varying.
        # Here we pad on patch dimension for batching.
        pixel_values = rnn_utils.pad_sequence(
            [features['pixel_values'] for features in features_list],
            batch_first=True,
            padding_value=0.0
        ).clone().detach()
        features['pixel_values'] = pixel_values

    if all('image_path_tensor' in f for f in features_list):
        image_path_tensor = rnn_utils.pad_sequence(
            [features['image_path_tensor'] for features in features_list],
            batch_first=True,
            padding_value=0
        ).clone().detach()
        features['image_path_tensor'] = image_path_tensor

    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    if getattr(agent, "checkpoint_path", None):
        logger.info("Initializing agent from checkpoint: %s", getattr(agent, "checkpoint_path", None))
        agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    #trainer = pl.Trainer(**cfg.trainer.params, callbacks=[pl.callbacks.ModelCheckpoint(monitor="val/loss_epoch",mode='min', save_top_k=5,every_n_epochs=1)])

    trainer_params = cfg.trainer.params
    latentsight_mode = str(getattr(cfg.agent, "latentsight_train_mode", "legacy")).replace("-", "_")
    #if latentsight_mode == "low_lr_full_finetune" and str(getattr(trainer_params, "strategy", "")) == "ddp":
        # Full/partial VLM finetuning can intentionally leave some modules unused
        # on a given step (for example optional world modulation branches or frozen
        # teacher/distillation-only paths). Lightning DDP otherwise raises
        # "parameters that were not used in producing the loss". Restrict the
        # slower unused-parameter detection to this memory-heavy mode instead of
        # changing the global default for all agents.
    #if latentsight_mode in ["low_lr_full_finetune", "vlm_lora"] and str(getattr(trainer_params, "strategy", "")) == "ddp":
    ddp_unused_param_modes = [
        "low_lr_full_finetune",
        "vlm_lora",
        "llm_planner_finetune",
        "freeze_vit_projector_train_llm_planner",
    ]
    if latentsight_mode in ddp_unused_param_modes and str(getattr(trainer_params, "strategy", "")) == "ddp":   
        
        trainer_params.strategy = "ddp_find_unused_parameters_true"
        logger.info("latentsight_train_mode=%s: using trainer strategy=%s", latentsight_mode, trainer_params.strategy)
    trainer = pl.Trainer(**trainer_params, callbacks=[pl.callbacks.ModelCheckpoint(monitor="val/loss_epoch",mode='min', save_top_k=5,every_n_epochs=1)])
    
    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
