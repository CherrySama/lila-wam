"""Measure real multi-view DINO + action-model training before the long run."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from dataloader.dataset import create_dataset, collate_fn
from models.model_runner import ModelFactory, VLAWrapper
from train import build_train_config_from_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/panthera_dual.yaml')
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    torch.set_num_threads(cfg.system.cpu_threads)
    torch.manual_seed(cfg.training.seed)
    device, dtype = torch.device('cuda'), torch.bfloat16
    encoder, dim, registers, patch = ModelFactory.create_vision_encoder(
        cfg.model.vision_encoder.checkpoint_path, dtype, device)
    action = ModelFactory.create_action_model(
        cfg, dim, len(cfg.model.vision_encoder.feat_layers), patch_size=patch).to(device)
    model = VLAWrapper(
        encoder, action, cfg.training.time_sampler, cfg.model.vision_encoder.feat_layers,
        cfg.model.vision_encoder.include_cls_register, registers, device, dtype,
        cfg.dataset.norm_stats_path, build_train_config_from_yaml(cfg),
        cfg.model.future_feat.target_layer, norm_stats_key=cfg.dataset.norm_stats_key)
    model.train()
    dataset = create_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(action.parameters(), lr=cfg.training.learning_rate)
    measurements = []
    for index, batch in enumerate(loader):
        if index == 3:
            break
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=dtype):
            loss, info = model(batch)
        loss.backward()
        assert torch.isfinite(loss), 'Non-finite loss'
        assert not encoder.training, 'Frozen DINO was switched into training mode'
        assert all(p.grad is None for p in encoder.parameters()), 'DINO must stay frozen'
        camera_grad = action.camera_embeddings.grad.abs().sum(dim=(0, 2, 3))
        assert (camera_grad > 0).all(), 'Both camera embeddings must receive gradients'
        torch.nn.utils.clip_grad_norm_(action.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        result = {'batch': index, 'batch_size': len(batch['state']),
                  'loss': loss.item(), 'flow_loss': info['loss_mse'],
                  'future_loss': info['loss_future_feat'],
                  'camera_gradient_sums': camera_grad.tolist(),
                  'seconds': time.perf_counter() - start,
                  'peak_memory_gib': torch.cuda.max_memory_allocated() / 2**30}
        print(json.dumps(result), flush=True)
        measurements.append(result)
    Path('log').mkdir(exist_ok=True)
    Path('log/panthera_preflight.json').write_text(json.dumps(measurements, indent=2) + '\n')


if __name__ == '__main__':
    main()
