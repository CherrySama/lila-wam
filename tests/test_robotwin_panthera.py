"""Single-arm simulation targets, camera order, and holdout invariants."""
import unittest

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from dataloader.dataset import create_dataset
from models.model_runner import ModelFactory


class SimulationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = OmegaConf.load('configs/panthera_robotwin.yaml')
        cls.train = create_dataset(cls.config)
        cls.val = create_dataset(cls.config, val=True)

    def test_episode_split_and_final_anchor_exclusion(self):
        self.assertEqual((len(self.train.episodes), len(self.val.episodes)), (270, 30))
        self.assertEqual(sum(len(ep['state']) for ep in self.train.all_episodes), 58722)
        self.assertEqual(len(self.train) + len(self.val), 58722 - 300)
        identities = lambda split: {(x['dataset'], x['episode_index']) for x in split}
        self.assertFalse(identities(self.train.split_manifest['train']) &
                         identities(self.val.split_manifest['validation']))

    def test_targets_start_after_current_state_and_do_not_cross_episodes(self):
        for dataset in [self.train, self.val]:
            episode = dataset.episodes[0]
            first = dataset[0]
            np.testing.assert_array_equal(first['state'][0], episode['state'][0])
            np.testing.assert_array_equal(first['action_sequence'][0], episode['state'][1])
            last = dataset[int(dataset.ends[0]) - 1]
            np.testing.assert_array_equal(last['state'][0], episode['state'][-2])
            np.testing.assert_array_equal(last['action_sequence'][0], episode['state'][-1])
            self.assertEqual(last['action_mask'].sum().item(), 1)
            self.assertFalse(last['future_mask'].item())

    def test_camera_cache_preserves_collector_rgb_order(self):
        episode = self.train.episodes[0]
        frames = np.load(self.train._cache_path(episode), mmap_mode='r')
        with h5py.File(episode['hdf5_path'], 'r') as stream:
            for index, camera in enumerate(self.config.dataset.camera_names):
                encoded = stream[f'observation/{camera}/rgb'][7]
                image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                expected = cv2.resize(image, tuple(self.config.dataset.image_size),
                                      interpolation=cv2.INTER_LINEAR)
                np.testing.assert_array_equal(frames[7, index], expected)

    def test_simulation_factory_uses_both_cameras(self):
        config = OmegaConf.create(OmegaConf.to_container(self.config))
        config.model.action_expert.hidden_size = 32
        config.model.action_expert.num_heads = 4
        config.model.action_expert.depth = 1
        config.model.action_expert.adapter_depth = 1
        config.model.action_expert.vlm_adapter_num_queries = 3
        config.model.future_feat.num_decoder_layers = 1
        model = ModelFactory.create_action_model(config, 16, 3)
        self.assertEqual(model.num_cameras, 2)
        self.assertEqual(model.future_feat_decoder.num_queries, 600)
        features = [torch.randn(1, 2, 6, 16) for _ in range(3)]
        output = model(torch.tensor([0.5]), torch.randn(1, 32, 7),
                       torch.randn(1, 1, 7), features)
        self.assertEqual(tuple(output['final_pred'].shape), (1, 32, 7))


if __name__ == '__main__':
    unittest.main()
