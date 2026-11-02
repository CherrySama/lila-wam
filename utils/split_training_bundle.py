"""Split the verified training archive into GitHub Release assets under 2 GiB."""
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'dist/LiLa-WAM-panthera-training.tar.gz'
    checksum = source.with_name(source.name + '.sha256')
    expected = checksum.read_text().split()[0]
    destination = root / 'dist/github-release'
    destination.mkdir(parents=True, exist_ok=True)
    aggregate = hashlib.sha256()
    checksums = []
    with source.open('rb') as stream:
        number = 0
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            name = f'{source.name}.part-{number:02d}'
            output = destination / name
            temporary = output.with_name(name + '.partial')
            digest = hashlib.sha256()
            size = 0
            with temporary.open('wb') as target:
                while block:
                    target.write(block)
                    digest.update(block)
                    aggregate.update(block)
                    size += len(block)
                    if size == 1024**3:
                        break
                    block = stream.read(min(8 * 1024 * 1024, 1024**3 - size))
            temporary.replace(output)
            checksums.append(f'{digest.hexdigest()}  {name}')
            print(f'Created {name}: {size} bytes', flush=True)
            number += 1
    if aggregate.hexdigest() != expected:
        raise ValueError('Input archive does not match its SHA256 checksum')
    (destination / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')
    (destination / checksum.name).write_text(checksum.read_text())
    print(f'Verified archive checksum; {len(checksums)} release parts ready', flush=True)


if __name__ == '__main__':
    main()
