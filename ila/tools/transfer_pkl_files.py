#!/usr/bin/env python3
if __name__ == '__main__':
    from iti.helpers import mllogger
    import os
    from os.path import join as pjoin
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_prefix", type=str,
                        help="sweep file")
    parser.add_argument("--dry_run", action='store_true', help="dry run")
    args = parser.parse_args()

    params_path = pjoin(args.snapshot_prefix, 'parameters.pkl')
    paths = mllogger.glob(params_path)

    # List paths found by grep
    print(f'{len(paths)} files are found:')
    for path in paths:
        print(path)

    target_prefix = os.getenv('SNAPSHOT_ROOT')
    assert target_prefix
    # Download each file and upload it to GCP
    skipped = []
    for i, path in enumerate(paths):
        target_path = target_prefix.rstrip('/') + '/' + path
        print(f'{i} / {len(paths)}: Downloading & Uploading', path)
        print('\ttarget path:', target_path)

        if not args.dry_run:
            from tempfile import NamedTemporaryFile
            if mllogger.glob(target_path):
                print('target_path already exists!!', target_path)
                print('skipping.')
                skipped.append(target_path)
                continue

            with NamedTemporaryFile(delete=True) as tfile:
                mllogger.download_file(path=path, to=tfile.name)
                mllogger.upload_file(tfile.name, target_path=target_path)

    print('skipped paths', skipped)
