"""Script to convert the CribsTV dataset to LMDB.

LMDBs (http://www.lmdb.tech/doc/) should provide faster loading & less load on the filesystem.

NOTE: This process takes quite a while!
Results are cached (i.e. LMDBs aren't recomputed unless forced) so the script can be interrupted and restarted.
"""
import shutil
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import src.devkits.cribs_tv as ctv
from src.external_libs import write_image_database
from src.paths import DATA_PATHS as PATHS


def process_dataset(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> None:
    """Process the entire CribsTV dataset."""

    # NOTE: Intrinsics are not exported because they are arbitrary and can simply be created on the fly.

    # Copy split files.
    if not (path := dst_dir/'splits').is_dir():
        shutil.copytree(src_dir/'splits', path)

    items = ctv.Item.load_split('train')
    items += ctv.Item.load_split('val')

    # Export all sequences.
    args = [(src_dir/seq, dst_dir/seq, [i for i in items if i.seq == seq], overwrite) for seq in ctv.get_seqs()]
    with Pool() as p: list(p.starmap(export_seq, args))
    # [export_seq(*arg) for arg in args]


def export_seq(src_dir: Path, dst_dir: Path, items: list[ctv.Item], overwrite: bool = False) -> None:
    """Export all scenes in a sequence as a single LMDB."""
    if not overwrite and dst_dir.is_dir():
        print(f"\t\t-> Images already exported")
        return

    items = {f'{i.scene}/{i.stem}' for i in items}

    print(f"\t\t-> Exporting images to '{dst_dir}'")
    files = sorted(src_dir.glob('*/*.png'))
    files = {f'{f.parent.stem}/{f.stem}': f for f in files}
    if items:
        files = {k: v for k, v in files.items() if k in items}
        assert len(files) == len(items), f"Missing files: {items - set(files.keys())}"

    write_image_database(files, dst_dir)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to convert the CribsTV dataset to LMDB.')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing LMDBs.')
    args = parser.parse_args()

    process_dataset(PATHS['cribs_tv'], PATHS['cribs_tv_lmdb'], overwrite=args.overwrite)
