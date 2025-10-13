import itertools
from dataclasses import dataclass, field
from typing import Iterable, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import multiprocessing as mp
from tqdm import tqdm

from .consts import *
from .loce_storage_helpers import all_pkls_to_npz, df_where_cols_equal, df_from_csv, df_to_csv
from .logging import log_info, log_warn
from ..background_pasting.background_pasting import PasteOnBackground, BGType
from ..data_structures.caching import RAMCache
from ..loce import LoCEActivationsTensorExtractor
from ..loce.loce_optimizer import LoCEOptimizationEngine


def plot_means(df: pd.DataFrame, values: Union[str, list[str]],  # ='iou',
               group_cols: Optional[Sequence[str]] = None,  # =(DATA, MODEL),
               compare: Optional[Sequence[str]] = None,  # =(DEPTH,),
               side_by_side: Optional[Sequence[str]] = None,  # =(CE_METHOD,),
               top_to_bottom: Optional[Sequence[str]] = None,
               restrict_to: Optional[dict[str, Any]] = None,  # {CE_METHOD: NET2VEC},
               pretty_names: dict = None,
               axsize: tuple[float, float] = (5,3),
               **plot_args,
               ):
    """Plot a (row of) bar chart(s) with mean and standard deviation of df[value]

    Args:
        df: the dataframe with values to plot
        values: the value to (average and) plot
        group_cols: the df column that should become the x-Axis values
        compare: for each x-axis value, plot bars for each value of the df column(s)
            value (combinations) in `compare` side-by-side;
            if the values are pivoted, i.e., not of the form
                  COMP   VAL
            idx1  comp1  val1
            idx2  comp2  val2
            idx3  comp1  val3
            idx4  comp2  val4

            but of the form

                  COMP1  COMP2
            idx1' val1   val2
            idx2' val3   val4
            ...

            use compare = None, values = ['COMP1', 'COMP2'] instead.

        side_by_side: for each value combination of columns in `side-by-side`, plot one new chart.
        restrict_to: before plotting, restrict the df to only those values fulfilling all these {col: value} constraints
        pretty_names: before plotting, rename all indices of index and columns according to this mapping.
            Mind that these must still be unique!
        figsize: width x height aspect ratio of a single axis plot inside the figure.
        **plot_args: arguments to DataFrame.plot()

    Returns: the plotted figure object

    """
    if len(df.index) == 0: raise ValueError("Received empty dataframe for plotting!")

    values: list[str] = [values] if isinstance(values, str) else list(values)
    restrict_to = restrict_to or {}
    side_by_side = side_by_side or []
    top_to_bottom = top_to_bottom or []
    compare = compare or []
    group_cols = group_cols or []
    group_cols = [c for c in group_cols if c not in restrict_to.keys()]
    pretty_names = pretty_names or {}

    # Restrict to data of interest
    if restrict_to:
        df = df_where_cols_equal(df, restrict_to)
        if len(df.index) == 0: raise ValueError(f"DataFrame empty after applying restrictions {restrict_to}")

    # list of restrictors to be applied per columns
    side_by_side_values: Iterable[tuple] = itertools.product(*[df[k].unique() for k in side_by_side])
    side_by_side_setts: list[dict] = [dict(zip(side_by_side, vals)) for vals in side_by_side_values]

    # list of restrictors to be applied per row
    top_to_bottom_values: Iterable[tuple] = itertools.product(*[df[k].unique() for k in top_to_bottom])
    top_to_bottom_setts: list[dict] = [dict(zip(top_to_bottom, vals)) for vals in top_to_bottom_values]


    fig, axes = plt.subplots(len(top_to_bottom_setts), len(side_by_side_setts),
                             figsize=(axsize[0] * len(side_by_side_setts), axsize[1] * len(top_to_bottom_setts)),
                             sharex=True, sharey=True, squeeze=False)
    for ax_row, top_to_bottom_vals in zip(axes, top_to_bottom_setts):
        for ax, side_by_side_vals in zip(ax_row, side_by_side_setts):
            # Summarize along the columns that are not to be directly compared:
            filtered = (
                # restrict to the values to be shown in this axis
                df_where_cols_equal(df, {**top_to_bottom_vals, **side_by_side_vals})
            )
            # Skip subselections that do not contain any element
            if len(filtered.index) == 0: continue
            grouped = (filtered
                # remove the respective columns (not needed for the rest)
                .drop(columns=[k for k in {*top_to_bottom_vals.keys(), *side_by_side_vals.keys()}])
                # aggregate a mean and std subcolumn for each value, setting all x-axis labels as index
                .pivot_table(values=sorted(values), index=sorted(group_cols + compare),
                             aggfunc={v: ['mean', 'std'] for v in values})
                # make x-axis labels again available for pivoting
                .reset_index()
                # pivot the compare columns: each compare vlaue will become a separate column, and thus a separate plot line
                .pivot(columns=compare, index=group_cols)
                )
            
            # some default plot naming
            # title = (f"Average {values[0] if len(values)==1 else ""} "
            #         f"({', '.join([f'{k}={v}' for k, v in side_by_side_vals.items()])})")
            ylabel = ", ".join([f"{pretty_names.get(v, v)}" for k, v in top_to_bottom_vals.items()])
            xlabel = ", ".join([f"{pretty_names.get(v, v)}" for k, v in side_by_side_vals.items()])

            reset_levels = 1 if len(values)>1 else [0,1]
            stds = grouped.loc[:, (slice(None), 'std')].T.reset_index(level=reset_levels, drop=True).T
            means = grouped.loc[:, (slice(None), 'mean')].T.reset_index(level=reset_levels, drop=True).T
            (means
             # add some pretty names
             .rename(columns=pretty_names, index=pretty_names)
             # now plot
             .plot(yerr=stds.rename(columns=pretty_names, index=pretty_names),
                   ax=ax,
                   **{'kind': 'bar', 'xlabel': xlabel, 'ylabel': ylabel, # 'title': title,
                   **plot_args})
            )
    return fig, means, stds


def to_cos_sim_matrix(d: pd.DataFrame, col_template: str = "{}") -> pd.DataFrame:
    """Receive a dataframe with columns 'bg_randomizer_key' (unique values) and 'ce',
    and return a dataframe with 'bg_randomizer_key' column and each a column {bg_randomizer_key value}."""
    bg_randomizer_keys: list[str] = d['bg_randomizer_key'].unique()
    nums_bg_per_ce: dict[str, list[int]] = {b: d[d['bg_randomizer_key'] == b]['num_bgs_per_ce'].unique() for b in
                                            bg_randomizer_keys}

    d: pd.DataFrame = d.set_index(['bg_randomizer_key', 'num_bgs_per_ce']).sort_index()

    cos_sim_on_cols = lambda b1, b2, n: cos_sim(d.loc[(b1, n if b1 != 'vanilla' else 1), 'ce'], d.loc[(b2, n), 'ce'])
    d_dict = {col_template.format(b1):
        {
            (b2, n): cos_sim_on_cols(b1, b2, n)
            for b2 in bg_randomizer_keys
            for n in nums_bg_per_ce[b2] if n in nums_bg_per_ce[b1] or b1 == 'vanilla'
        }
        for b1 in bg_randomizer_keys
    }
    return pd.DataFrame(d_dict).reset_index(names=['bg_randomizer_key', 'num_bgs_per_ce'])

def to_cos_sims(df: pd.DataFrame, settings_cols: list[str]) -> pd.DataFrame:
    """Calc and add {bg_randomizer_key value} columns to df."""
    # discard additional infos:
    cos_sims_glo = df[[*settings_cols, 'ce']]
    # group into variations of BG and num samples of BG:
    cos_sims_glo = cos_sims_glo.groupby(
        by=[s for s in settings_cols if s not in ('bg_randomizer_key', 'num_bgs_per_ce')], group_keys=True)
    # calculate cosine sims between each combi of (BG1, n), (BG2, n) (some wrapping here, since 'vanilla' should always use n=1)
    cos_sims_glo = cos_sims_glo.apply(to_cos_sim_matrix, include_groups=False)
    # get rid of the grouping index
    cos_sims_glo = cos_sims_glo.reset_index()
    # display(cos_sims_glo)
    return cos_sims_glo


def plot_cos_sim(df: pd.DataFrame,
                 settings_cols: list[str] = ('run', 'ce_method', 'bg_randomizer_key', 'num_bgs_per_ce', 'dataset_key',
                                             'model_key', 'category_id', 'layer')
                 ) -> pd.DataFrame:
    """Calculate and plot cosine similarities between pairs of (background1, num_imgs per bg1), (background2, num_imgs per bg2).
    All bg+num combis are compared against ('vanilla', 1)."""
    cos_sims_glo = to_cos_sims(df, settings_cols)

    # Plot side-by-side
    plot_means(cos_sims_glo, values=[*df[BG].unique()],
               group_cols=[BG], side_by_side=[NUM_BG])
    return cos_sims_glo


def get_projections_torch(loces: list[np.ndarray],
                          acts: torch.Tensor,
                          device=('cuda' if torch.cuda.is_available() else 'cpu'),
                          ) -> torch.Tensor:
    """
    Get projection of LoCE and activations

    Args:
        loces (np.ndarray): LoCE of shape BxC
        acts (torch.Tensor): (single) activations of shape 1xCxHxW

    Kwargs:
        downscale_factor (float = None): downscale factor

    Returns:
        (np.ndarray) np.uint8 image array
    """
    loces_np = np.stack(loces)  # shape: B x C
    acts = torch.as_tensor(acts, device=device, dtype=torch.float)  # shape:  1 x C x H x W

    loce3d = torch.as_tensor(loces_np, device=device, dtype=torch.float
                             ).reshape(*loces_np.shape, 1, 1)  # shape: B x C x 1 x 1
    loce3d = loce3d.nan_to_num(1000)  # assume NaN values are representing simply large params

    # multiply and sum along the channel dimension
    projection = (acts * loce3d).sum(dim=1)

    # normalize
    projection = torch.sigmoid(projection)

    return projection


def get_jaccard_indices(projections: torch.Tensor, masks: torch.Tensor,
                        thresh_proj: float = 0.5, thresh_masks: float = 0., eps=0.0001):
    projections = projections.gt(thresh_proj).reshape(projections.size()[0], -1)
    masks = masks.gt(thresh_masks).reshape(masks.size()[0], -1)

    intersect = torch.logical_and(masks, projections)
    union = torch.logical_or(masks, projections)

    iou = (intersect.sum(dim=-1) + eps) / (union.sum(dim=-1) + eps)

    return iou


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays."""
    a_l2norm: float = np.linalg.norm(a, ord=2)
    b_l2norm: float = np.linalg.norm(b, ord=2)
    if np.allclose(a, 0.) and np.allclose(b, 0.): return 1.
    if np.allclose(a, 0.) or np.allclose(b, 0.): return 0.
    return np.dot(a, b) / (a_l2norm * b_l2norm)


def possibly_int(v: Union[str, int]) -> Union[str, int]:
    """Try to convert to int, else stick to string."""
    try:
        return int(v)
    except ValueError:
        return str(v)


def with_globalized_ces(orig_df, experiment_setting_cols: Sequence[str], add_depth=False):
    experiment_setting_cols = list(experiment_setting_cols)
    # remove any 'ignore' category entries accidentally created
    df = orig_df.copy()

    # some initial cleansing
    df = df[df[CAT] != 'ignore']

    _is_global = df[CE_METHOD].isin(GLOBAL_CE_METHODS)

    ## Global: deduplicate, global vectors are the same on all img_ids
    df_glo = df[_is_global].drop_duplicates(subset=experiment_setting_cols)

    ## Local
    df_loc = df[~_is_global]

    ## local to global: average and mark globalization
    df_loglo = df_loc[[*experiment_setting_cols, 'ce']].groupby(by=experiment_setting_cols)[['ce']].mean().reset_index()
    df_loglo[CE_METHOD] = df_loglo[CE_METHOD].apply("globalized_{}".format)

    # This script only evaluates global concept methods
    df = pd.concat([df_glo, df_loglo, df_loc],
                   keys=['global', 'local_to_global', 'local'],
                   names=['global_or_local', 'index_to_remove'],
                   sort=False
                   ).reset_index().drop(columns=['index_to_remove'])

    # add some derived meta info about layers
    if add_depth:
        df[DEPTH] = to_layer_depth_vals(df)

    return df

def to_layer_depth_vals(df: pd.DataFrame) -> pd.Series:
    def get_depth(row: pd.Series):
        for i, depth_key in enumerate(["early", "middle", "late"]):
            if any((row[MODEL] == m and row[LAYER] == ls[i] for m, ls in LAYERS_BY_MODEL.items())):
                return depth_key
        return 'unknown'

    return df.apply(get_depth, axis='columns')

@dataclass
class IoUEvalConfig:
    results_dir: str = field(default='./results')
    output_dir: str = field(default=None)
    cache_dir: str = field(default='./data/results/predictions')
    npz_dir: str = field(default='./data/results/npz')
    force_update_npz: bool = field(default=False, metadata=dict(type=bool, action="store_true"))
    device: Optional[Union[str, torch.device]] = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata=dict(type=set_device))
    num_bgs_per_test_img: int = field(default=4)
    target_shape: tuple[int, int] = field(
        default=(80, 80),
        metadata=dict(nargs=2))
    """the common shape to resize activations and masks to"""
    verbose: bool = field(default=0, metadata=dict(type=int, action="count"))
    dataset_keys: list[str] = field(default=None, metadata=dict(nargs='+'))

    # Experiment settings for subselecting
    run: str = field(default='*')
    ce_method: str = field(default='*')
    bg_randomizer_key: str = field(default='*')
    num_bgs_per_ce: str = field(default='*')
    dataset_key: str = field(default='*')
    model_key: str = field(default='*')
    category_id: str = field(default='.+')
    img_id: str = field(default='.+')
    layer: str = field(default='.+')

    def to_selection_specs(self) -> str:
        return dict(
            run=self.run,
            ce_method=self.ce_method,
            bg_randomizer_key=self.bg_randomizer_key,
            num_bgs_per_ce=self.num_bgs_per_ce,
            dataset_key=self.dataset_key,
            model_key=self.model_key,
            category_id=self.category_id,
            img_id=self.img_id,
            layer=self.layer,
        )

    def get_output_path(self) -> str:
        return os.path.join(self.output_dir, 'results.csv') if self.output_dir else None


async def calc_ious_global(config: IoUEvalConfig) -> tuple[pd.DataFrame, list[str]]:
    print(f"Starting for config {config}")
    if config.cache_dir:
        print("INFO: caching to directory {}".format(config.cache_dir))
        log_info("INFO: caching to directory {}".format(config.cache_dir))

    to_cache_path = lambda ds_key, cat_id, bg_key: os.path.join(config.cache_dir, ds_key, bg_key, f"results_{cat_id}.csv")

    # load all concept embeddings + meta-info
    ce_infos = await all_pkls_to_npz(results_dir=config.results_dir, npz_dir=config.npz_dir,
                                     force_update=config.force_update_npz, verbose=config.verbose > 0,
                                     **config.to_selection_specs())

    # to dataframe
    orig_df = pd.DataFrame(ce_infos)

    df = with_globalized_ces(orig_df, experiment_setting_cols=EXPERIMENT_SETTING_COLS)
    df = df[df[GLO_OR_LOC] != 'local']

    # results collectors
    all_results = []
    all_errors = []

    with torch.no_grad():
        # what to iterate over
        data_concept_combis = [(d, c) for d in df[DATA].unique()
                               for c in df[df[DATA] == d][CAT].unique()
                               if config.dataset_keys is None or d in config.dataset_keys]
        background_id_types = {'vanilla': None,
                               **PLACES_SUBSETS,
                               'any': [b for bs in PLACES_SUBSETS.values() for b in bs]}.items()
        layers_by_model = {model_key: df[df[MODEL] == model_key]['layer'].unique()
                           for model_key in df[MODEL].unique()}

        if config.verbose == 1:
            data_concept_combis = tqdm(data_concept_combis)
        for i, (dataset_key, category_id) in enumerate(data_concept_combis, 1):
            if config.verbose == 1: data_concept_combis.set_description(f"{dataset_key}, {category_id}")
            if config.verbose > 1: log_info(f"Starting {dataset_key}, {category_id} ({i}/{len(data_concept_combis)})")

            # Build the test dataset
            dataset_builder = TEST_DATA_BUILDERS[dataset_key]
            test_dataset: SegmentationDataset = dataset_builder(category_ids=[possibly_int(category_id)],
                                                                device=config.device)
            test_dataset = RAMCache(test_dataset)

            # Iterate over the different background types
            if config.verbose == 1: background_id_types = tqdm(background_id_types, leave=False)
            for b, (background_key, test_bg_ids) in enumerate(background_id_types):

                # if existing, load from cache
                cache_path = to_cache_path(ds_key=dataset_key, cat_id=category_id, bg_key=background_key) \
                    if config.cache_dir is not None else None
                if cache_path is not None and os.path.isfile(cache_path):
                    results_df = df_from_csv(cache_path)
                    all_results.extend(results_df.apply(lambda row: row.to_dict(), axis=1))
                    continue
                results = []  # the results to be collected for this sub-run
                errors = []

                if config.verbose == 1: background_id_types.set_description(f"Evaluating backgrounds {background_key}")
                if config.verbose > 1:
                    log_info(f"Starting {background_key=} ({b}/{len(background_id_types)})")

                # Create and set the bg dataset
                test_dataset.clear_cache()
                assert test_dataset.cache == {}
                match background_key:
                    case 'vanilla':
                        test_dataset.transform = None
                    case _:
                        ## Prepare the background randomizer
                        bg_dataset: SegmentationDataset = BG_DATA_BUILDERS['places'](
                            device=config.device, category_ids=test_bg_ids)
                        forbidden_classes = [imagenet_c for cname in test_dataset.cat_name_by_id.values()
                                             for imagenet_c in AS_IMAGENET_IDS_OR_NAMES.get(cname, [])] or None
                        bg_paster: PasteOnBackground = PasteOnBackground(
                            background_loader=bg_dataset, bg_type=BGType.full_bg,
                            num_imgs=config.num_bgs_per_test_img,
                            forbidden_classes_bg=forbidden_classes, )

                        # set the transform (also clears cache if needed)
                        test_dataset.transform = bg_paster

                # Loop over models & layers
                layer_loop = layers_by_model.items()
                if config.verbose == 1: layer_loop = tqdm(layer_loop, leave=False)
                for j, (model_key, layers) in enumerate(layer_loop, 1):
                    if config.verbose == 1: layer_loop.set_description(f"Evaluating {model_key}")
                    if config.verbose > 1:
                        log_info(f"Starting {model_key=} ({j}/{len(layer_loop)})")

                    # Get the model
                    model_builder = WRAPPED_MODELS[model_key]
                    propagator, processor = model_builder(layers, device=config.device)
                    activations_extractor = LoCEActivationsTensorExtractor(propagator, model_key, processor=processor)

                    # Loop over images, calculating IoU for each image separately
                    test_img_ids = test_dataset.img_ids
                    if config.verbose == 1: test_img_ids = tqdm(test_img_ids, desc="Images processed", leave=False)
                    for k, test_img_id in enumerate(test_img_ids, 1):

                        # get batch of images
                        img_pil, seg_mask = test_dataset[test_img_id]
                        if isinstance(img_pil, (tuple, list)):  # non-vanilla case
                            assert background_key != 'vanilla'
                            imgs_pil, seg_masks = list(img_pil), [seg_mask]*len(img_pil)
                        else:  # vanilla case
                            assert background_key == 'vanilla'
                            imgs_pil, seg_masks = [img_pil], [seg_mask]

                        # flatten imgs and masks, obtain activations, and resize both activations and seg masks to the same target_shape
                        activations, seg_masks_reshaped = LoCEOptimizationEngine._get_reshaped_activations_segmentations(
                            imgs_pil, seg_masks,
                            target_shape=config.target_shape,
                            activations_extractor=activations_extractor,
                            device=config.device)

                        # Now loop over layers
                        for layer, acts in activations.items():

                            # get the global concept vectors for this subselection
                            curr_df = df_where_cols_equal(df, {
                                DATA: dataset_key, CAT: category_id, MODEL: model_key, LAYER: layer,
                            })
                            if len(curr_df.index) == 0:
                                warn_text = f"No concept embeddings for {dataset_key=}, {category_id=}, {model_key=}, {layer=}"
                                errors.append(f"WARNING: {warn_text}")
                                if config.verbose > 1: log_warn(warn_text)
                                continue
                            if config.verbose > 1:
                                log_info(f"Calculating "
                                         f"{dataset_key=}/{category_id=} ({i}/{len(data_concept_combis)}, "
                                         f"{model_key=}@{layer=} ({j}/{len(layer_loop)}), "
                                         f"{test_img_id=} ({k}/{len(test_img_ids)})")

                            # extract the concept embeddings and metadata
                            ces, infos = zip(*[(row.pop('ce'), row.to_dict()) for _, row in curr_df.iterrows()])

                            # iterate over every activation in that batch (i.e., every bg randomized version of img_pil)
                            for img_count, (act, seg_mask_reshaped) in enumerate(zip(acts, seg_masks_reshaped, strict=True)):
                                act: torch.Tensor = act.unsqueeze(0)

                                # get the predictions
                                projections: torch.Tensor = get_projections_torch(ces, act, device=config.device)

                                # get the IoU score
                                ious = get_jaccard_indices(projections, seg_mask_reshaped)

                                # store infos
                                for info, iou in zip(infos, ious, strict=True):
                                    # add metadata + IoU
                                    d = dict(**{**info,
                                             IOU: iou.cpu().item(),
                                             TEST_IMG_ID: test_img_id, TEST_IMG_SUBID: img_count,
                                             TEST_BG: background_key})
                                    results.append(d)

                                # TODO: store masks and projections?

                all_results.extend(results)
                all_errors.extend(errors)

                # cache
                if cache_path is not None:
                    results_df = pd.DataFrame(results)
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    df_to_csv(results_df, cache_path)
                    print(f"Intermediate results for {dataset_key=}, {category_id=}, {background_key=} stored in {cache_path}")
                    log_info(f"Intermediate results for {dataset_key=}, {category_id=}, {background_key=} stored in {cache_path}")
                    for err in errors: print(err)

    all_results_df = pd.DataFrame(all_results)
    return all_results_df, all_errors


def set_device(device: Optional[str] = None, gpu_id: Optional[int] = None) -> 'torch.device':
    """Get the default torch device."""
    # Set the GPU ID here and check if the specified GPU is available.
    if device is None:
        if torch.cuda.is_available():
            if gpu_id is not None and gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')
            else:
                device = torch.device('cuda')
            mp.set_start_method('spawn')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device("cpu")
    torch.set_default_device(device)
    return device
