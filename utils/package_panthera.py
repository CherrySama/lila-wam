"""Build and verify a portable project + real/sim data + DINO training archive."""
import argparse
import hashlib
import os
import tarfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.git', '.venv', '.cache', 'dist', 'log', 'checkpoints_vla',
           'eval_result', '__pycache__'}


def project_files():
    for directory, folders, names in os.walk(ROOT, followlinks=False):
        folders[:] = sorted(name for name in folders if name not in EXCLUDE)
        for folder in folders:
            if (Path(directory) / folder).is_symlink():
                raise ValueError(f'Unexpected symbolic directory: {directory}/{folder}')
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix == '.pyc':
                continue
            if path.is_symlink():
                raise ValueError(f'Unexpected symbolic file: {path}')
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dist/LiLa-WAM-panthera-training.tar.gz')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'dist'):
        raise ValueError('Transfer archives must be saved inside the excluded dist directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    paths = list(project_files())
    expected = {}
    print(f'Packaging {len(paths)} files, {sum(p.stat().st_size for p in paths)} bytes', flush=True)
    last_update = time.monotonic()
    try:
        with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
            for index, path in enumerate(paths, 1):
                name = 'LiLa-WAM/' + path.relative_to(ROOT).as_posix()
                with path.open('rb') as stream:
                    expected[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
                archive.add(path, arcname=name, recursive=False)
                if time.monotonic() - last_update > 15:
                    print(f'Packed {index}/{len(paths)} files', flush=True)
                    last_update = time.monotonic()
        print('Verifying every archive member against its source SHA256', flush=True)
        seen = set()
        with tarfile.open(temporary, 'r|gz') as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                with archive.extractfile(member) as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != expected[member.name]:
                    raise ValueError(f'Archive checksum mismatch: {member.name}')
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError('Archive is incomplete')
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    checksum = output.with_name(output.name + '.sha256')
    checksum.write_text(f'{digest}  {output.name}\n')
    print(f'Verified {len(paths)} files; {output.stat().st_size} archive bytes\n'
          f'{output}\nSHA256: {digest}', flush=True)


if __name__ == '__main__':
    main()
