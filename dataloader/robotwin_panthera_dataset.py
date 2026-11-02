"""Panthera single-arm RoboTwin HDF5 data, with joint-state future targets."""
import hashlib
import json
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from .lerobot_dataset import LeRobotDataset


logger = logging.getLogger(__name__)


class RobotWinPantheraDataset(LeRobotDataset):
    """Share episode-safe chunk sampling with the real-data adapter."""

    def _load_episodes(self):
        fraction = self.config.dataset.validation_fraction
        if not 0 < fraction < 1:
            raise ValueError('validation_fraction must be between 0 and 1')
        if min(self.action_offsets) < 1:
            raise ValueError('RoboTwin stores observed qpos; action offsets must start at +1 or later')
        self.feature_names = {key: [f'joint{i}.pos' for i in range(1, 7)] + ['gripper.pos']
                              for key in ['action', 'observation.state']}
        self.split_manifest = {'seed': self.config.dataset.split_seed, 'train': [], 'validation': []}
        for value in self.config.dataset.dataset_dir:
            root = Path(value).resolve()
            manifest = json.loads((root / 'collection_manifest.json').read_text())
            files = sorted((root / 'data').glob('episode*.hdf5'),
                           key=lambda path: int(path.stem.removeprefix('episode')))
            if len(files) < 2:
                raise ValueError(f'Need at least two HDF5 episodes: {root}')
            ids = [int(path.stem.removeprefix('episode')) for path in files]
            if len(set(ids)) != len(ids):
                raise ValueError(f'Duplicate episode IDs: {root}')
            rng = np.random.default_rng(self.config.dataset.split_seed)
            count = min(len(files) - 1, max(1, round(len(files) * fraction)))
            validation_ids = set(rng.permutation(ids)[:count].tolist())
            for path, episode_id in zip(files, ids):
                record = manifest['episodes'].get(str(episode_id))
                if not record or not record['hdf5']['passed'] or not record['replay_validation']['passed']:
                    raise ValueError(f'Episode lacks successful collection/replay validation: {path}')
                with h5py.File(path, 'r') as stream:
                    if stream.attrs.get('schema_version') != 'panthera-single-v1':
                        raise ValueError(f'Expected Panthera single-arm schema: {path}')
                    if stream.attrs.get('collection_fingerprint') != manifest['provenance']['fingerprint']:
                        raise ValueError(f'Collection fingerprint differs: {path}')
                    if int(stream.attrs.get('collection_episode', -1)) != episode_id:
                        raise ValueError(f'Episode ID differs: {path}')
                    state = stream['joint_action/vector'][:].astype(np.float32)
                    length = len(state)
                    if state.shape != (length, 7) or length < 2 or not np.isfinite(state).all():
                        raise ValueError(f'Invalid joint-state sequence: {path}')
                    if self.config.common.action_dim != 7 or self.config.common.state_dim != 7:
                        raise ValueError('Panthera simulation requires 7D state and action')
                    for camera in self.camera_names:
                        if stream[f'observation/{camera}/rgb'].shape != (length,):
                            raise ValueError(f'Unsynchronized camera {camera}: {path}')
                episode = {'root': root, 'hdf5_path': path,
                           'metadata': {'episode_index': episode_id},
                           'state': state, 'action': state, 'sample_length': length - 1,
                           'expected_sha256': record['hdf5']['sha256']}
                is_val = episode_id in validation_ids
                split = 'validation' if is_val else 'train'
                self.split_manifest[split].append({'dataset': Path(value).as_posix(),
                                                  'episode_index': episode_id})
                self.all_episodes.append(episode)
                if is_val == self.val:
                    self.episodes.append(episode)
        if not self.episodes:
            raise ValueError('Dataset split is empty')

    def _cache_path(self, episode):
        path = episode['hdf5_path']
        settings = json.dumps([str(path), self.camera_names, self.image_size])
        digest = hashlib.sha256(settings.encode()).hexdigest()[:16]
        return self.cache_dir / f'{episode["root"].name}-{path.stem}-{digest}.npy'

    def prepare_cache(self):
        """Decode both cameras once; verify each HDF5 against its collection hash."""
        cv2.setNumThreads(1)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for number, episode in enumerate(self.all_episodes, 1):
            path = episode['hdf5_path']
            cache = self._cache_path(episode)
            signature = {'source': str(path), 'size': path.stat().st_size,
                         'mtime_ns': path.stat().st_mtime_ns,
                         'sha256': episode['expected_sha256'],
                         'cameras': self.camera_names, 'image_size': list(self.image_size)}
            shape = (len(episode['state']), len(self.camera_names),
                     self.image_size[1], self.image_size[0], 3)
            sidecar = cache.with_suffix('.json')
            with path.open('rb') as stream:
                actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual_hash != episode['expected_sha256']:
                raise ValueError(f'HDF5 differs from collection manifest: {path}')
            if cache.exists() and sidecar.exists() and json.loads(sidecar.read_text()) == signature:
                images = np.load(cache, mmap_mode='r')
                if images.shape == shape:
                    continue
            temporary = cache.with_suffix('.partial.npy')
            images = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=shape)
            try:
                with h5py.File(path, 'r') as stream:
                    for view, camera in enumerate(self.camera_names):
                        for index, encoded in enumerate(stream[f'observation/{camera}/rgb']):
                            # The collector passes RGB arrays directly to cv2.imencode;
                            # cv2.imdecode restores that original array order (no channel swap).
                            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if image is None:
                                raise ValueError(f'Invalid JPEG: {path}, camera={camera}, frame={index}')
                            images[index, view] = cv2.resize(image, self.image_size,
                                                           interpolation=cv2.INTER_LINEAR)
                images.flush()
                del images
                temporary.replace(cache)
                sidecar.write_text(json.dumps(signature, indent=2) + '\n')
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            if number % 25 == 0 or number == len(self.all_episodes):
                logger.info('Prepared simulation RGB cache: %d/%d episodes',
                            number, len(self.all_episodes))

    def _pixels(self, episode, local_index):
        cache = self._cache_path(episode)
        if cache not in self._frames:
            if not cache.exists():
                raise FileNotFoundError(f'Run utils/prepare_panthera.py with the simulation config: {cache}')
            self._frames[cache] = np.load(cache, mmap_mode='r')
        image = self._frames[cache][local_index].astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32))
        image /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(image.transpose(0, 3, 1, 2)))
