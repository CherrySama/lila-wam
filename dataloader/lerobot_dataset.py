"""Episode-safe LeRobot v3 sampling for synchronized multi-camera training."""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


def video_cache_paths(root, info, meta, camera, image_size, cache_dir):
    key = f'observation.images.{camera}'
    video = root / info['video_path'].format(
        video_key=key, chunk_index=meta[f'videos/{key}/chunk_index'],
        file_index=meta[f'videos/{key}/file_index'])
    digest = hashlib.sha256(str(video).encode()).hexdigest()[:16]
    width, height = image_size
    cache = Path(cache_dir) / f'{root.name}-{camera}-{digest}-{width}x{height}.npy'
    return video, cache


def prepare_video_cache(dataset):
    """Decode AV1 once into RGB memory maps; validate every video timestamp."""
    import av
    import cv2
    cv2.setNumThreads(1)
    dataset.cache_dir.mkdir(parents=True, exist_ok=True)
    videos = {}
    for ep in dataset.all_episodes:
        for camera in dataset.camera_names:
            video, cache = video_cache_paths(ep['root'], ep['info'], ep['metadata'], camera,
                                             dataset.image_size, dataset.cache_dir)
            fps = ep['info']['fps']
            key = f'videos/observation.images.{camera}'
            start = ep['metadata'][f'{key}/from_timestamp']
            end = ep['metadata'][f'{key}/to_timestamp']
            if abs((end - start) * fps - len(ep['action'])) > 0.05:
                raise ValueError(f'{video}: episode/video durations differ')
            count = round(end * fps)
            if video not in videos:
                videos[video] = {'cache': cache, 'fps': fps, 'count': count}
            else:
                videos[video]['count'] = max(videos[video]['count'], count)
    for video, item in videos.items():
        cache, fps, count = item['cache'], item['fps'], item['count']
        signature = {'source': str(video), 'size': video.stat().st_size,
                     'mtime_ns': video.stat().st_mtime_ns, 'image_size': list(dataset.image_size),
                     'frames': count, 'fps': fps, 'color': 'RGB'}
        sidecar = cache.with_suffix('.json')
        if cache.exists() and sidecar.exists() and json.loads(sidecar.read_text()) == signature:
            frames = np.load(cache, mmap_mode='r')
            if frames.shape == (count, dataset.image_size[1], dataset.image_size[0], 3):
                logger.info('Using validated frame cache: %s', cache)
                continue
        logger.info('Decoding %s (%d frames) -> %s', video, count, cache)
        temporary = cache.with_suffix('.partial.npy')
        frames = np.lib.format.open_memmap(
            temporary, mode='w+', dtype=np.uint8,
            shape=(count, dataset.image_size[1], dataset.image_size[0], 3))
        decoded = 0
        try:
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'
                stream.codec_context.thread_count = 4
                for index, frame in enumerate(container.decode(stream)):
                    if index >= count or frame.pts is None:
                        raise ValueError(f'{video}: unexpected frame count or missing PTS')
                    timestamp = float(frame.pts * frame.time_base)
                    if abs(timestamp - index / fps) > 0.5 / fps:
                        raise ValueError(f'{video}: frame {index} is not synchronized ({timestamp})')
                    rgb = frame.to_ndarray(format='rgb24')
                    frames[index] = cv2.resize(rgb, dataset.image_size, interpolation=cv2.INTER_LINEAR)
                    decoded += 1
            if decoded != count:
                raise ValueError(f'{video}: decoded {decoded} frames, expected {count}')
            frames.flush()
            del frames
            temporary.replace(cache)
            sidecar.write_text(json.dumps(signature, indent=2) + '\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class LeRobotDataset(Dataset):
    def __init__(self, config, val=False):
        self.config = config
        self.val = val
        self.camera_names = list(config.dataset.camera_names)
        self.image_size = tuple(config.dataset.image_size)
        if any(size % 16 for size in self.image_size):
            raise ValueError('Image dimensions must be multiples of the DINO patch size (16)')
        if list(config.dataset.indices_config.camera_indices) != [0]:
            raise ValueError('LeRobot multi-view sampling currently uses the current frame only')
        if len(config.dataset.indices_config.state_indices) != config.common.proprio_len:
            raise ValueError('State history length must match proprio_len')
        if len(config.dataset.indices_config.action_indices) != config.common.action_chunk_size:
            raise ValueError('Action offsets must match action_chunk_size')
        if config.model.use_task_cond or config.dataset.image_aug:
            raise ValueError('This single-task adapter uses no task vector or image augmentation')
        self.cache_dir = Path(config.dataset.cache_dir).resolve()
        self.state_offsets = np.asarray(config.dataset.indices_config.state_indices)
        self.action_offsets = np.asarray(config.dataset.indices_config.action_indices)
        self.future_offset = config.dataset.get('future_frame_offset')
        if self.future_offset is None:
            self.future_offset = config.common.action_chunk_size
        if self.future_offset < 0:
            raise ValueError('Future frame offset must be non-negative')
        self.use_future_feat = config.model.future_feat.enabled
        self.episodes = []
        self.all_episodes = []
        self._frames = {}
        self._load_episodes()
        lengths = [ep.get('sample_length', len(ep['action'])) for ep in self.episodes]
        self.ends = np.cumsum(lengths)
        logger.info('%s %s: %d episodes, %d samples, cameras=%s',
                    config.dataset.type, 'validation' if val else 'train',
                    len(lengths), len(self), self.camera_names)

    def _load_episodes(self):
        """Read numeric arrays and split complete episodes reproducibly per source."""
        seed = self.config.dataset.split_seed
        fraction = self.config.dataset.validation_fraction
        if not 0 < fraction < 1:
            raise ValueError('validation_fraction must be between 0 and 1')
        expected_names = None
        self.split_manifest = {'seed': seed, 'train': [], 'validation': []}
        for root_value in self.config.dataset.dataset_dir:
            root = Path(root_value).resolve()
            info = json.loads((root / 'meta/info.json').read_text())
            if info['codebase_version'] != 'v3.0':
                raise ValueError(f'Expected LeRobot v3.0: {root}')
            names = {key: info['features'][key]['names'] for key in ['action', 'observation.state']}
            if expected_names is not None and names != expected_names:
                raise ValueError(f'Joint order differs between datasets: {root}')
            expected_names = names
            self.feature_names = names
            for key, dim in [('action', self.config.common.action_dim),
                             ('observation.state', self.config.common.state_dim)]:
                if info['features'][key]['shape'] != [dim]:
                    raise ValueError(f'{root}: incompatible {key} dimension')
            for cam in self.camera_names:
                if f'observation.images.{cam}' not in info['features']:
                    raise ValueError(f'{root}: missing camera {cam}')
            if info['fps'] != self.config.dataset.fps:
                raise ValueError(f'{root}: sampling frequency differs from configuration')
            metadata = []
            for path in sorted((root / 'meta/episodes').rglob('*.parquet')):
                table = pq.read_table(path)
                columns = [c for c in table.column_names if not c.startswith('stats/')]
                metadata.extend(table.select(columns).to_pylist())
            metadata.sort(key=lambda ep: ep['episode_index'])
            if len(metadata) < 2:
                raise ValueError(f'Need at least two episodes per source: {root}')
            ids = [ep['episode_index'] for ep in metadata]
            if len(set(ids)) != len(ids) or len(ids) != info['total_episodes']:
                raise ValueError(f'{root}: inconsistent episode metadata')
            rng = np.random.default_rng(seed)
            count = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
            validation_ids = set(rng.permutation(ids)[:count].tolist())
            tables = {}
            for meta in metadata:
                chunk, file = meta['data/chunk_index'], meta['data/file_index']
                path = root / info['data_path'].format(chunk_index=chunk, file_index=file)
                if path not in tables:
                    tables[path] = pq.read_table(path).to_pandas()
                rows = tables[path]
                rows = rows[rows.episode_index == meta['episode_index']].sort_values('frame_index')
                length = int(meta['length'])
                if len(rows) != length or not np.array_equal(rows.frame_index, np.arange(length)):
                    raise ValueError(f'{root}: incomplete episode {meta["episode_index"]}')
                timestamp = rows.timestamp.to_numpy(dtype=np.float64)
                if not np.allclose(timestamp, np.arange(length) / info['fps'], atol=1e-4):
                    raise ValueError(f'{root}: irregular timestamps in episode {meta["episode_index"]}')
                episode = {'root': root, 'metadata': meta, 'info': info,
                           'action': np.stack(rows.action).astype(np.float32),
                           'state': np.stack(rows['observation.state']).astype(np.float32),
                           'timestamp': timestamp}
                if not all(np.isfinite(episode[k]).all() for k in ['action', 'state']):
                    raise ValueError(f'{root}: non-finite joint data')
                is_val = meta['episode_index'] in validation_ids
                split = 'validation' if is_val else 'train'
                # Preserve relative IDs so saved splits survive moving the project.
                identity = {'dataset': Path(root_value).as_posix(),
                            'episode_index': meta['episode_index']}
                self.split_manifest[split].append(identity)
                self.all_episodes.append(episode)
                if is_val == self.val:
                    self.episodes.append(episode)
        if not self.episodes:
            raise ValueError('Dataset split is empty')

    def normalization_stats(self):
        """Fit statistics exclusively on the training trajectories."""
        if self.val:
            raise ValueError('Normalization must be fitted on the training split')
        result = {'feature_names': self.feature_names, 'split': self.split_manifest, 'panthera': {}}
        for key in ['action', 'state']:
            values = np.concatenate([ep[key] for ep in self.episodes])
            result['panthera'][key] = {
                'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist(),
                'mean': values.mean(axis=0).tolist(), 'std': values.std(axis=0).tolist(),
            }
        return result

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        return self._get_sample(index)

    def _pixels(self, episode, local_index):
        views = []
        for camera in self.camera_names:
            video, cache = video_cache_paths(
                episode['root'], episode['info'], episode['metadata'], camera,
                self.image_size, self.cache_dir)
            if cache not in self._frames:
                if not cache.exists():
                    raise FileNotFoundError(f'Run utils/prepare_panthera.py first: {cache}')
                self._frames[cache] = np.load(cache, mmap_mode='r')
            offset = episode['metadata'][f'videos/observation.images.{camera}/from_timestamp']
            frame_index = round((offset + episode['timestamp'][local_index]) * episode['info']['fps'])
            image = self._frames[cache][frame_index].astype(np.float32) / 255.0
            image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32))
            image /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
            views.append(np.transpose(image, (2, 0, 1)))
        return torch.from_numpy(np.ascontiguousarray(np.stack(views)))

    def _get_sample(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.ends, index, side='right'))
        episode = self.episodes[episode_index]
        anchor = index - (int(self.ends[episode_index - 1]) if episode_index else 0)
        length = len(episode['action'])
        actions = anchor + self.action_offsets
        states = np.clip(anchor + self.state_offsets, 0, length - 1)
        result = {
            'pixel_values': self._pixels(episode, anchor),
            'state': torch.from_numpy(episode['state'][states].copy()),
            'action_sequence': torch.from_numpy(episode['action'][np.clip(actions, 0, length - 1)].copy()),
            'action_mask': torch.from_numpy((actions >= 0) & (actions < length)),
        }
        if self.use_future_feat:
            future = anchor + self.future_offset
            result['future_pixel_values'] = self._pixels(episode, min(length - 1, future))
            result['future_mask'] = torch.tensor(future < length, dtype=torch.bool)
        return result
