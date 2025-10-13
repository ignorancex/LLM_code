import asyncio
import functools
import pickle
import re
import signal
from dataclasses import dataclass, fields
from pathlib import Path
from string import Formatter
from typing import Iterable, Any, Sequence

import aiofiles
import numpy as np
import pandas as pd
from pandas.api.types import is_array_like
from tqdm import tqdm

from .consts import *
from .files import read_pickle
from .logging import log_warn
from ..loce.loce import LoCEMultilayerStorage, LoCE


## CE LOADING
@dataclass(kw_only=True)
class LoCEMetaInfo:
    run: str
    ce_method: str
    bg_randomizer_key: str
    num_bgs_per_ce: int
    dataset_key: str
    model_key: str
    category_id: Union[str, int]
    img_id: str
    layer: str = None
    loss: float = None
    rel_path: str = None
    abs_path: str = None
    ce: np.ndarray = None

    def get_multilayer_loce(self) -> LoCEMultilayerStorage:
        return read_pickle(self.abs_path)

    def get_loce(self) -> Optional[LoCE]:
        if self.layer is None:
            return None
        loces: LoCEMultilayerStorage = self.get_multilayer_loce()
        return loces.get_loce(self.layer)


async def get_all_ces(results_root: str, cache: str = None,
                      **selectors) -> pd.DataFrame:
    # load from cache if possible:
    if not selectors and cache is not None and os.path.isfile(cache):
        df = pd.read_csv(cache).infer_objects()
        df.loc[:, 'ce'] = pd.eval(df.ce)
        df.loc[:, 'ce'] = df.ce.apply(np.array)
        return df

    all_multilayer_meta_infos = await get_ce_pkl_paths(results_root, **selectors)

    # Get individuals per layer
    batch_size = 500  # process that many files in parallel
    batches = (all_multilayer_meta_infos[i:min(len(all_multilayer_meta_infos), i + batch_size)]
               for i in range(0, len(all_multilayer_meta_infos) + batch_size, batch_size))
    all_meta_infos = []
    for batch in batches:
        tasks = []
        for task in (asyncio.ensure_future(_load_ce_infos_from_multistorageloce(meta_info))
                     for meta_info in batch):
            tasks.append(task)
        all_meta_infos.extend([i for infos in await asyncio.gather(*tasks) for i in infos])

    df = pd.DataFrame(all_meta_infos)

    # If really ALL are loaded, cache the results:
    if not selectors and cache is not None:
        df_to_store = df.copy()
        df_to_store.ce = df.ce.apply(lambda x: x.tolist())
        df_to_store.to_csv(cache, index=False)
    return df


async def get_ce_pkl_paths(results_root: Union[str, Path],
                           *,
                           run=None,
                           ce_method: '_CE_METHOD_KEY' = None,
                           bg_randomizer_key: '_BG_DATASET_KEY' = None,
                           num_bgs_per_ce: int = None,
                           dataset_key: '_DATASET_KEY' = None,
                           model_key: '_MODEL_KEY' = None,
                           category_id: Union[int, str] = None,
                           img_id: Union[int, str] = None) -> list[LoCEMetaInfo]:
    """Get a """
    selection_specs = dict(run=run, ce_method=ce_method, bg_randomizer_key=bg_randomizer_key,
                           num_bgs_per_ce=num_bgs_per_ce,
                           dataset_key=dataset_key, model_key=model_key, category_id=category_id, img_id=img_id)
    # if not, load multilayer loces from directories
    results_dir: Path = Path(results_root)
    assert os.path.isdir(results_dir), f"Given results_dir is not a directory: {results_dir.absolute()}"
    glob_template = STORAGE_TEMPLATE
    glob_pattern: str = glob_template.format(**{k: (s or '*') for k, s in selection_specs.items()})
    match_host_pattern: str = glob_template.format(
        **{k: (s or f'(?P<{k}>[^/]+)') for k, s in selection_specs.items()}
    ).replace(".", r"\.").replace('**', r'[^/]+').replace("*", "[^/]+")
    infos: list[LoCEMetaInfo] = []
    for pkl_path in results_dir.glob(glob_pattern):
        infos.append(LoCEMetaInfo(**meta_data_from_path(match_host_pattern, pkl_path, results_dir)))

    # Filter net2vec duplicates
    if ce_method is not None and 'net2vec' not in ce_method:
        return infos

    # Remove all net2vec related meta info items and filter them to only contain unique entries:
    net2vec_infos: list[LoCEMetaInfo] = [infos.pop(i) for i, info in enumerate(infos) if 'net2vec' in info.ce_method]
    pbar = tqdm(total=len(net2vec_infos), desc="Net2Vecs processed")
    while len(net2vec_infos) > 0:
        # Add first occurrence of vector to list
        meta_info = net2vec_infos.pop(0)
        infos.append(meta_info)

        # Remove duplicates
        idx = 0
        while idx < len(net2vec_infos):
            info = net2vec_infos[idx]
            # is duplicate?
            _field_names_to_compare: Iterable[str] = (f.name for f in fields(meta_info) if
                                                      'path' not in (k := f.name) and k != 'img_id')
            if all((getattr(info, k) == getattr(meta_info, k) for k in _field_names_to_compare)):
                net2vec_infos.pop(idx)
            else:
                idx += 1

        # # Validate that these truly are duplicates
        # curr_loces: LoCEMultilayerStorage = meta_info.get_multilayer_loce()
        # for dup in duplicates:
        #     dup_loces: LoCEMultilayerStorage = dup.get_multilayer_loce()
        #     for layer, dup_loce in dup_loces.loce_storage.items():
        #         assert np.allclose(curr_loces.get_loce(layer).loce, dup_loce.loce), \
        #             f"WARNING: non-equal duplicate in layer {layer}: {dup.abs_path} != {meta_info_dict.abs_path}"
        pbar.update(n=1)
    pbar.close()
    infos.extend(net2vec_infos)
    return infos


def meta_data_from_path(pattern, pkl_path: Path, results_dir) -> dict:
    # unravel the paths into metadata
    path: str = str(pkl_path.relative_to(results_dir)) if pkl_path is not None else results_dir
    match = re.search(pattern, path)
    if match is None:
        raise ValueError(f"WARNING: failed to parse {pkl_path.relative_to(results_dir)} with {pattern}")
    meta_info_dict: dict[str, Union[int, str]] = match.groupdict()
    if 'num_bgs_per_ce' in meta_info_dict:
        meta_info_dict['num_bgs_per_ce'] = int(meta_info_dict['num_bgs_per_ce'])
    if pkl_path is not None:
        meta_info_dict |= dict(rel_path=str(pkl_path.relative_to(results_dir)), abs_path=str(pkl_path.absolute()))
    return meta_info_dict


async def get_ce(abs_path: str, layer: str, ce: np.ndarray = None, **_) -> np.ndarray:
    """Load a concept embedding from given (multi-ce) file if ce is not given, else return ce."""
    if ce is not None:
        return ce
    loces = await _load_multistorageloce(abs_path)
    return loces.loce_storage[layer].loce


async def _load_multistorageloce(abs_path: Union[Path, str]) -> LoCEMultilayerStorage:
    """Load a concept embedding from given (multi-ce) file."""
    async with aiofiles.open(abs_path, 'rb') as f:
        content = await f.read()
    loces: LoCEMultilayerStorage = pickle.loads(content)
    return loces


async def _load_ce_infos_from_multistorageloce(meta_info: dict[str, Union[int, str]]) -> list[dict]:
    """Load a list of dicts with meta information for all concept embeddings stored in meta_info['abs_path']."""
    loces: LoCEMultilayerStorage = await _load_multistorageloce(meta_info['abs_path'])

    assert str(loces.segmentation_category_id) == str(meta_info['category_id']), \
        f"Concept mismatch: expected {meta_info['category_id']}, got {loces.segmentation_category_id}"
    return [meta_info | dict(layer=layer, loss=loce.loss, ce=loce.loce)
            for layer, loce in loces.loce_storage.items()]


async def pkl_to_ce_dict(pkl_file: Union[Path, str]) -> dict[str, np.ndarray]:
    """Load concept vectors from pkl_file and return a dict {concept/img_id/layer: vector}."""
    try:
        loces = await _load_multistorageloce(pkl_file)
    except Exception as e:
        print(f"ERROR: failed to open pkl file {pkl_file}: {e}")
        e.add_note(f"pkl file: {pkl_file}")
        raise e
    ce_by_layer = {layer: loce.loce for layer, loce in loces.loce_storage.items()}

    cat_id, img_id = os.path.basename(pkl_file).replace(".pkl", "").split('_', maxsplit=1)
    ces = {f"{cat_id}/{img_id}/{layer}": ce for layer, ce in ce_by_layer.items()}

    return ces


def ask_exit(signame, loop):
    """Helper func to handle exit signals to asyncio processes."""
    print("got signal %s: exit" % signame)
    loop.stop()


async def all_pkls_to_npz(results_dir: str, npz_dir: str = None, force_update: bool = False, verbose: bool = True,
                          category_id='.+', img_id='.+', layer='.+',  # non-folder selection specs
                          **selection_specs) -> list[LoCEMetaInfo]:
    # some asyncio stuff: add handles for SIGINT and SIGTERM signals.
    loop = asyncio.get_running_loop()

    for signame in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signame),
            functools.partial(ask_exit, signame, loop))

    glob_selection_specs = dict(run='*', ce_method='*', bg_randomizer_key='*', num_bgs_per_ce='*', dataset_key='*',
                                model_key='*') | selection_specs
    # get all folders that match the selection specs
    glob_pattern: str = os.path.dirname(STORAGE_TEMPLATE).format(**glob_selection_specs)
    pkl_folders: Sequence[Path] = list(Path(results_dir).glob(glob_pattern))
    if verbose: pkl_folders: tqdm = tqdm(pkl_folders)

    # iterate all available folders
    all_ces = []
    for pkl_subfolder in pkl_folders:
        pkl_relfolder = Path(pkl_subfolder).relative_to(results_dir)
        if verbose: pkl_folders.set_description(str(pkl_relfolder))
        ces_sublist: list = await pkls_to_npz_with_metainfo(results_dir, pkl_relfolder, npz_dir, force_update,
                                                            category_id=category_id, img_id=img_id, layer=layer)
        all_ces.extend(ces_sublist)

    if len(all_ces) == 0:
        print(f"WARNING: no pkl files found in {results_dir}")
    return all_ces


async def pkls_to_npz_with_metainfo(results_dir: str, pkl_relfolder: Union[str, Path], npz_dir: str = None,
                                    force_update: bool = False,
                                    category_id='.+', img_id='.+', layer='.+', # further non-folder selection specs
                                    **selection_specs) -> list[LoCEMetaInfo]:
    # actual loading
    ce_dict, npz_path = await pkls_to_npz(results_dir, pkl_relfolder, npz_dir, force_update=force_update)
    if ce_dict == {}: return []

    # basic meta info given by selection specs
    meta_info = dict(selection_specs)

    # parse further meta info from folder name
    match_keys: list[str] = [i[1] for i in Formatter().parse(os.path.dirname(STORAGE_TEMPLATE)) if i[1] is not None]
    match_host_pattern: str = os.path.dirname(STORAGE_TEMPLATE).format(
        **{k: selection_specs.get(k, f'(?P<{k}>[^/]+)') for k in match_keys}
    ).replace(".", r"\.").replace('**', r'[^/]+').replace("*", "[^/]+")
    meta_info |= meta_data_from_path(match_host_pattern, Path(npz_path or os.path.join(results_dir, pkl_relfolder)),
                                     npz_dir)

    # wrap in info objects
    ces = []
    for cat_img_layer, ce in ce_dict.items():
        if re.match(f"{category_id}/{img_id}/{layer}", cat_img_layer):
            cat_id, i_id, l = cat_img_layer.split('/')
            ces.append(LoCEMetaInfo(**meta_info, category_id=cat_id, img_id=i_id, layer=l, ce=ce))
    return ces


def load_npz(path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Evil hack to make this async."""
    try:
        with np.load(path) as npz_handle:
            # Make sure to read out the file contents here:
            npz = dict(**npz_handle)
        return npz
    # If something goes wrong, add more debug info
    except Exception as e:
        e.add_note(f"Failed to load npz file {path}: {e}")
        raise e


def to_pkl_file_glob(run='run1', ce_method='*', bg_randomizer_key='*', num_bgs_per_ce='*', dataset_key='*',
                     model_key='*', category_id='*', img_id='*', layer='*', ):
    return os.path.join(run, ce_method, bg_randomizer_key, f"{num_bgs_per_ce}_bgs_per_ce", dataset_key,
                        f"loce_*_{model_key}", f"{category_id}_{img_id}.pkl")


def pkl_todo_count(results_dir, *,
                   run, ce_method, dataset_key, bg_randomizer_key, num_bgs_per_ce, category_id, model_key,
                   total_img_counts: dict[str, int] = None
                   ):
    """Hacky implementation to check whether there are still files missing for an experiment."""
    total_img_counts = total_img_counts or {
            **{dataset_key: 0},  # the default: not in experiments.
            **{dataset_key: len(ids[category_id]['train'])
               for dataset_key, ids in DATA_SPLITS.items() if category_id in ids},
            **{'imagenets50': 10}  # the ImageNetS50 default
    }

    _file_glob = to_pkl_file_glob(run=run, ce_method=ce_method, bg_randomizer_key=bg_randomizer_key,
                                  num_bgs_per_ce=num_bgs_per_ce, dataset_key=dataset_key, model_key=model_key,
                                  category_id=category_id)
    _existing_files_in_out_dir = list(Path(results_dir).glob(_file_glob))

    to_do_count = 1 if 'net2vec' in ce_method else total_img_counts[dataset_key]

    return to_do_count - len(_existing_files_in_out_dir)


async def pkls_to_npz(results_dir: str, pkl_rel_subfolder: str,
                      npz_folder: str = None,
                      force_update: bool = False,
                      npz_filename_templ: str = 'all_ce_count_{count}',
                      ) -> tuple[dict[str, np.ndarray], str]:
    """Load all LoCE pkl files in subfolder and store concept vectors in one .npz file.
    Format of npz dict: see pkl_to_ce_dict

    Return: Dict of all content written to npz file; npz file path
    """
    pkl_subfolder: Path = Path(os.path.join(results_dir, pkl_rel_subfolder))
    pkl_files: list[Path] = list(pkl_subfolder.glob('*.pkl'))

    # set up npz saving/loading
    npz_subfolder = None
    if npz_folder is not None:
        # ensure the new subfolder exists
        npz_subfolder = os.path.join(npz_folder, pkl_rel_subfolder)
        os.makedirs(npz_subfolder, exist_ok=True)

        # load from existing .npz file if existing and matching the desired count ...
        npz_glob_pattern: str = npz_filename_templ.format(count=len(pkl_files)) + '.npz'
        existing_npzs = list(Path(npz_subfolder).glob(npz_glob_pattern))
        if not force_update and len(existing_npzs) > 0:
            if len(existing_npzs) > 1:
                raise FileNotFoundError(
                    f'Expected one but found {len(existing_npzs)} instances of files {npz_glob_pattern}: {existing_npzs}')
            # ... then load and return:
            existing_npz = existing_npzs[0]
            try:
                npz: dict[str, np.ndarray] = load_npz(existing_npz)
                return npz, str(existing_npz)
            except Exception as e:
                log_warn(f'Failed to load {existing_npz} (recreating): {e}')

    # iterate through pkl files in subfolder (batch wise)
    ce_dict = {}

    batch_size = 300  # the number of files that may be open simultaneously
    batches = (pkl_files[i: min(len(pkl_files), i + batch_size)]
               for i in range(0, len(pkl_files) + batch_size, batch_size))
    for batch in batches:
        tasks = [asyncio.create_task(pkl_to_ce_dict(pkl_file), name=str(pkl_file)) for pkl_file in batch]

        curr_ce_dicts = await asyncio.gather(*tasks, return_exceptions=True)
        # some error handling
        for i, ce_dict_or_exception in enumerate(curr_ce_dicts):
            if isinstance(ce_dict_or_exception, Exception):
                log_warn(f"Encountered exception processing pkl files (skipping file): {ce_dict_or_exception}")
                curr_ce_dicts.pop(i)

        ce_dict |= {k: v for ced in curr_ce_dicts for k, v in ced.items()}

    npz_path = None
    if npz_subfolder is not None:
        npz_path = os.path.join(npz_subfolder, npz_filename_templ.format(count=len(pkl_files)))
        with Path(npz_path).open('wb') as npz_file:
            try:
                np.savez(npz_file, **ce_dict)
            except Exception as e:
                log_warn(f'Failed to save {npz_path} (ignoring): {e}')

    npz_path: str = str(npz_path) + '.npz' if npz_path is not None else None
    # log_info(npz_path)
    return ce_dict, npz_path


def df_to_csv(df: pd.DataFrame, csv_path: str,
              np_cols: list[str] = ('ce', 'ce_baseline'),
              index: bool = False):
    df_ = df.copy()

    # convert all numpy vectors into lists to make them parsable later on:
    for np_col in (c for c in np_cols if c in df_.columns and is_array_like(df_[c])):
        df_.loc[:, np_col] = df_[np_col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    # now save
    df_.to_csv(csv_path, index=index, na_rep='NaN')


def df_from_csv(csv_path: str, np_cols: list[str] = ('ce', 'ce_baseline')) -> pd.DataFrame:
    # read from csv
    plain_df = pd.read_csv(csv_path)

    # unpack the numpy arrays
    for np_col in (c for c in np_cols if c in plain_df.columns):
        plain_df.loc[:, np_col] = plain_df[np_col].apply(lambda x: np.array(pd.eval(x)))

    return plain_df


def df_cols_equal(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Return the pandas Series which rows match the settings."""
    equal_per_col: pd.DataFrame = (df[list(settings.keys())] == pd.Series(settings))
    return equal_per_col.all(axis=1)


def df_where_cols_equal(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    return df[df_cols_equal(df, settings)]

