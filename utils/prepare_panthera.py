"""Prepare LeRobot or single-arm RoboTwin caches and training-only normalization."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from dataloader.dataset import create_dataset
from dataloader.lerobot_dataset import prepare_video_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/panthera_dual.yaml')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = OmegaConf.load(args.config)
    dataset = create_dataset(config)
    stats = dataset.normalization_stats()
    path = Path(config.dataset.norm_stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + '\n')
    logging.info('Saved training-only statistics and split: %s', path)
    if hasattr(dataset, 'prepare_cache'):
        dataset.prepare_cache()
    else:
        prepare_video_cache(dataset)
    val = create_dataset(config, val=True)
    for split, data in [('train', dataset), ('validation', val)]:
        for episode_index in range(len(data.episodes)):
            first = 0 if episode_index == 0 else int(data.ends[episode_index - 1])
            for index in [first, int(data.ends[episode_index]) - 1]:
                sample = data[index]
                width, height = config.dataset.image_size
                assert tuple(sample['pixel_values'].shape) == (
                    len(config.dataset.camera_names), 3, height, width)
                assert sample['action_sequence'].shape == (
                    config.common.action_chunk_size, config.common.action_dim)
        logging.info('%s: validated first/last samples in every episode (%d samples)', split, len(data))


if __name__ == '__main__':
    main()
