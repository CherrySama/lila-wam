"""Behavioral invariants for camera fusion, padding masks, and episode splits."""
import unittest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dataloader.lerobot_dataset import LeRobotDataset
from models.vla_model_fm import VLAModel, calc_flow_matching_loss, future_feature_cosine_loss


class CameraFusionTests(unittest.TestCase):
    def test_both_cameras_receive_gradients_and_keep_their_identity(self):
        torch.manual_seed(7)
        model = VLAModel(
            action_dim=7, proprio_dim=7, hidden_dim=32, num_heads=4, depth=2,
            action_len=4, dino_feat_dims=(16, 16), vlm_num_queries=3,
            adapter_depth=1, fusion_mode='concat', concat_out_dim=16,
            num_cameras=2, use_future_feat=True, future_feat_num_queries=12,
            future_feat_out_dim=16, future_feat_depth=1)
        # adaLN starts at zero; open its gates to test attention paths immediately.
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
        features = [torch.randn(2, 2, 6, 16, requires_grad=True) for _ in range(2)]
        actions, state = torch.randn(2, 4, 7), torch.randn(2, 1, 7)
        t = torch.tensor([0.3, 0.7])
        output = model(t, actions, state, features)
        future = model.future_feat_decoder(output['cond_tokens'])
        self.assertEqual(tuple(future.shape), (2, 12, 16))
        (output['final_pred'].square().mean() + future.square().mean()).backward()
        for features_at_layer in features:
            for camera in range(2):
                self.assertGreater(features_at_layer.grad[:, camera].abs().sum().item(), 0)
        self.assertGreater(model.camera_embeddings.grad.abs().sum().item(), 0)
        swapped = model(t, actions, state, [f.flip(1) for f in features])['final_pred']
        self.assertFalse(torch.allclose(output['final_pred'], swapped, atol=1e-7))

    def test_single_camera_checkpoint_structure_remains_compatible(self):
        model = VLAModel(hidden_dim=32, num_heads=4, depth=1, dino_feat_dims=(16,),
                         vlm_num_queries=3, adapter_depth=1)
        self.assertNotIn('camera_embeddings', model.state_dict())


class LossMaskTests(unittest.TestCase):
    def test_padded_actions_are_excluded(self):
        class ZeroModel(torch.nn.Module):
            def forward(self, t, noisy_actions, **kwargs):
                return {'final_pred': torch.zeros_like(noisy_actions), 'cond_tokens': None}
        actions = torch.tensor([[[2., 2.], [100., 100.]]])
        mask = torch.tensor([[True, False]])
        with patch('torch.randn_like', side_effect=torch.zeros_like):
            loss, _ = calc_flow_matching_loss(
                ZeroModel(), actions, None, None, action_mask=mask)
        self.assertAlmostEqual(loss.item(), 4.)

    def test_invalid_future_samples_do_not_affect_loss(self):
        pred = torch.tensor([[[1., 0.]], [[1., 0.]]], requires_grad=True)
        target = torch.tensor([[[1., 0.]], [[-1., 0.]]])
        loss = future_feature_cosine_loss(pred, target, torch.tensor([True, False]))
        self.assertAlmostEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(pred.grad[1].abs().sum().item(), 0.)
        self.assertEqual(future_feature_cosine_loss(pred, target, torch.zeros(2)).item(), 0.)


class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = OmegaConf.load('configs/panthera_dual.yaml')
        cls.train = LeRobotDataset(config)
        cls.val = LeRobotDataset(config, val=True)

    def test_split_and_normalization_have_no_validation_leakage(self):
        train_ids = {(x['dataset'], x['episode_index']) for x in self.train.split_manifest['train']}
        val_ids = {(x['dataset'], x['episode_index']) for x in self.val.split_manifest['validation']}
        self.assertFalse(train_ids & val_ids)
        self.assertEqual((len(train_ids), len(val_ids)), (45, 5))
        self.assertEqual(len(self.train) + len(self.val), 22646)
        expected = np.concatenate([ep['action'] for ep in self.train.episodes])
        stats = self.train.normalization_stats()['panthera']['action']
        np.testing.assert_allclose(stats['min'], expected.min(axis=0))
        np.testing.assert_allclose(stats['max'], expected.max(axis=0))

    def test_chunks_never_cross_episode_boundaries(self):
        for dataset in [self.train, self.val]:
            for ep, end in zip(dataset.episodes, dataset.ends):
                sample = dataset[int(end) - 1]
                self.assertEqual(sample['action_mask'].sum().item(), 1)
                self.assertFalse(sample['future_mask'].item())
                np.testing.assert_array_equal(sample['action_sequence'][0], ep['action'][-1])
                np.testing.assert_array_equal(sample['action_sequence'][-1], ep['action'][-1])
                self.assertEqual(tuple(sample['pixel_values'].shape), (2, 3, 240, 320))

    def test_split_and_statistics_survive_project_relocation(self):
        config = OmegaConf.load('configs/panthera_dual.yaml')
        original = self.train.normalization_stats()
        working_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            for relative in config.dataset.dataset_dir:
                source = working_directory / relative
                destination = Path(temporary) / relative
                shutil.copytree(source / 'meta', destination / 'meta')
                shutil.copytree(source / 'data', destination / 'data')
            try:
                os.chdir(temporary)
                moved = LeRobotDataset(config)
                self.assertEqual(original, moved.normalization_stats())
                self.assertTrue(all(not Path(x['dataset']).is_absolute()
                                    for split in ['train', 'validation']
                                    for x in moved.split_manifest[split]))
            finally:
                os.chdir(working_directory)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
