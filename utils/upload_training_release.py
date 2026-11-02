"""Finish/resume the private training Release and publish only verified assets."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'CherrySama/lila-wam'
TAG = 'panthera-training-data-v1'
STATUS = ROOT / 'log/github-upload-status.json'


def save_status(state, **values):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {'state': state, 'repository': REPOSITORY, 'tag': TAG,
               'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **values}
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(payload), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gh', default=shutil.which('gh'))
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not args.gh:
        raise ValueError('GitHub CLI is required')

    def gh(*arguments, timeout=120):
        return subprocess.check_output([args.gh, *arguments], text=True,
                                       stderr=subprocess.STDOUT, timeout=timeout)

    def api(path):
        return json.loads(gh('api', path))

    def check_private():
        repository = api(f'repos/{REPOSITORY}')
        if not repository['private'] or not repository['permissions']['push']:
            raise ValueError('Uploads require the specified private repository and write access')

    check_private()
    releases = api(f'repos/{REPOSITORY}/releases')
    release = next(item for item in releases if item['tag_name'] == TAG)
    release_id = release['id']
    directory = ROOT / 'dist/github-release'
    expected = {}
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if Path(name).name != name:
            raise ValueError('Checksum entries must be plain filenames')
        expected[name] = digest
    for name in ['SHA256SUMS', 'LiLa-WAM-panthera-training.tar.gz.sha256']:
        expected[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()

    def verified(asset):
        name = asset['name']
        return (name in expected and asset['state'] == 'uploaded'
                and asset['size'] == (directory / name).stat().st_size
                and asset.get('digest') == 'sha256:' + expected[name])

    def snapshot():
        current = api(f'repos/{REPOSITORY}/releases/{release_id}')
        assets = {item['name']: item for item in current['assets']}
        good = sorted(name for name, asset in assets.items() if verified(asset))
        save_status('uploading', verified_assets=good, expected_assets=len(expected),
                    waiting_for_pid=args.wait_pid)
        return current, assets

    current, assets = snapshot()
    if args.check_only:
        return
    if args.wait_pid:
        process = Path(f'/proc/{args.wait_pid}/cmdline')
        while process.exists():
            try:
                command = process.read_bytes().decode(errors='replace')
            except FileNotFoundError:
                break
            if 'release\x00upload' not in command or TAG not in command:
                break
            time.sleep(30)
            current, assets = snapshot()

    for name in expected:
        current, assets = snapshot()
        if name in assets and verified(assets[name]):
            continue
        for attempt in range(3):
            check_private()
            save_status('retrying_upload', asset=name, attempt=attempt + 1)
            try:
                gh('release', 'upload', TAG, str(directory / name), '--repo', REPOSITORY,
                   '--clobber', timeout=7200)
                current, assets = snapshot()
                if verified(assets[name]):
                    break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                print(f'Upload attempt failed for {name}: {error}', flush=True)
            time.sleep(15)
        else:
            raise RuntimeError(f'Could not upload and verify {name}')
    current, assets = snapshot()
    if not all(name in assets and verified(assets[name]) for name in expected):
        raise RuntimeError('Release assets are incomplete or have incorrect hashes')
    check_private()
    gh('api', '--method', 'PATCH', f'repos/{REPOSITORY}/releases/{release_id}',
       '-F', 'draft=false', '-f', 'make_latest=true')
    published = api(f'repos/{REPOSITORY}/releases/{release_id}')
    if published['draft']:
        raise RuntimeError('Release remains a draft')
    save_status('complete', url=published['html_url'], verified_assets=sorted(expected),
                expected_assets=len(expected))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save_status('failed', error=str(error))
        raise
