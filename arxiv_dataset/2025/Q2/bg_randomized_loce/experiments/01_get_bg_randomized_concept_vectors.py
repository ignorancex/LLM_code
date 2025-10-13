"""For now experiments are for the Pascal VOC 2012 dataset as foreground."""
import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime

import argparse_dataclass
from tqdm import tqdm

from bg_randomized_loce.background_pasting import PasteOnBackground, BGType
from bg_randomized_loce.data_structures.caching import RAMCache
from bg_randomized_loce.loce import LoCEActivationsTensorExtractor, TorchCustomLoCEBatchOptimizer
from bg_randomized_loce.loce.loce_optimizer import LoCEOptimizationEngine, Net2VecOptimizationEngine
from bg_randomized_loce.utils.consts import _BG_DATASET_KEY, _CE_METHOD_KEY, _MODEL_KEY, _DATASET_KEY
from bg_randomized_loce.utils.eval_util import set_device
from bg_randomized_loce.utils.loce_storage_helpers import pkl_todo_count, to_pkl_file_glob
from bg_randomized_loce.utils.logging import init_logger, log_info, log_msg, get_current_git_hash

# ================================
# OPTIONS
# ================================

OBJECTIVE_TYPES: dict[str, str] = {
    "loce_proper_bce": "proper_bce",
    "loce": "bce",
    "net2vec_proper_bce": "proper_bce",
    "net2vec": "bce",
}
assert set(OBJECTIVE_TYPES.keys()) == set(get_args(_CE_METHOD_KEY))

# ================================
# EXPERIMENT SETTINGS
# ================================

@dataclass(frozen=True, kw_only=True)
class BgRandExperimentConfig:
    """Experiment settings."""
    dataset_keys: list[str] = field(
        default=tuple(get_args(_DATASET_KEY)),
        metadata=dict(nargs='+'))
    model_keys: list[str] = field(
        default=tuple(get_args(_MODEL_KEY)),  # "efficientnet", "vit", "yolo", "detr"]
        metadata=dict(nargs='+'))
    concept_names: list[str] = field(
        default=tuple([*MSCOCOSegmentationDataset.ALL_CAT_NAMES_BY_ID.values(),
                       *ImageNetS50SegmentationDataset.ALL_CAT_NAMES_BY_ID.values()]),
        # ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe')
        metadata=dict(nargs='+'))
    category_ids: list[str] = field(
        default=None,
        metadata=dict(nargs='+'))
    bg_randomizer_keys: list[str] = field(
        default=tuple(get_args(_BG_DATASET_KEY)),
        metadata=dict(nargs='+'))
    output_dir: str = field(default="./results/run1")
    n_imgs_per_category: int = field(default=50)  # how many images are considered per concept
    nums_bgs_per_ce: list[int] = field(
        default=(1, 4, 8, 32),
        metadata=dict(nargs='+',
                      help="Number of background images per LoCE / image."))
    batch_size: Optional[int] = field(default=512)
    epochs: int = field(default=30)
    """How often a concept embedding should have seen each image."""
    target_shape: tuple[int, int] = field(
        default=(80, 80),
        metadata=dict(nargs=2))  # the common shape to resize activations and masks to
    ce_methods: list[str] = field(
        default=tuple(get_args(_CE_METHOD_KEY)),
        metadata=dict(nargs='+'))
    device: Optional[Union[str, torch.device]] = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata=dict(type=set_device))
    no_in_mem_caching: bool = field(default=True, metadata=dict(type=bool, action="store_false"))
    force_folder_descent: bool = field(default=False, metadata=dict(type=bool, action="store_true"))
    """Whether to force a descent into the respective folders
    (checks whether results exist then are only on img_id level)."""


if __name__ == "__main__":
    # Read config
    _parser = argparse_dataclass.ArgumentParser(BgRandExperimentConfig)
    config: BgRandExperimentConfig = _parser.parse_args()

    # Some initial logging
    init_logger(file_name=os.path.join(config.output_dir, "logs", f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
                log_level=logging.INFO)
    log_info(f"Git Commit Hash: {get_current_git_hash()}")
    log_info(f"Starting with config: {config}")

    bg_datasets: dict[str, SegmentationDataset] = {}

    data_concept_combinations = tqdm([(d, c_id, c_name) for d in config.dataset_keys
                                      for c_id, c_name in DATA_BUILDERS[d][0]
                                      if c_id != 'ignore' and (not config.category_ids or str(c_id) in config.category_ids)])
    for count1, (dataset_key, concept_id, concept_name) in enumerate(data_concept_combinations, 1):
        data_concept_combinations.set_description(f"{dataset_key}/{concept_name} "
                                                  f"({count1}/{len(data_concept_combinations)})")

        # Populate the vanilla datasets
        _, dataset_builder = DATA_BUILDERS[dataset_key]

        # Create, validate, and add the dataset
        vanilla_concept_dataset: SegmentationDataset = dataset_builder(category_ids=[concept_id],
                                                                       device=config.device)
        if len(vanilla_concept_dataset) == 0:
            log_msg(f"WARNING: No suitable image-mask-pairs for {concept_name=}, {dataset_key=}.", logging.WARNING)
            continue
        vanilla_concept_dataset.shuffle()
        if not config.no_in_mem_caching:
            vanilla_concept_dataset = RAMCache(vanilla_concept_dataset)

        # Prepare the next print
        randomizer_combinations = tqdm(
            [(bgr, n) for bgr, n in itertools.product(config.bg_randomizer_keys, config.nums_bgs_per_ce)
             if not (bgr == "vanilla" and n > 1)], leave=False)
        for count2, (bg_randomizer_key, num_bgs_per_ce) in enumerate(randomizer_combinations, 1):
            randomizer_combinations.set_description(f"{dataset_key}/{concept_id} "
                                                    f"({count1}/{len(data_concept_combinations)}), "
                                                    f"{bg_randomizer_key} w/ #imgs/bg={num_bgs_per_ce} "
                                                    f"({count2}/{len(randomizer_combinations)})")

            ## Prepare the background randomizer
            bg_dataset: SegmentationDataset = bg_datasets.setdefault(bg_randomizer_key,
                                                                     BG_DATA_BUILDERS[bg_randomizer_key](
                                                                         device=config.device))
            # Set the background type (or continue)
            if bg_randomizer_key == "vanilla":
                vanilla_concept_dataset.transform = None
            else:
                bg_type = BGType.full_bg if bg_randomizer_key in ["synthetic", "places"] else BGType.voronoi

                # Init the background paster transformation and set it for the dataset
                forbidden_classes = [imagenet_c for cname in vanilla_concept_dataset.cat_name_by_id.values()
                                     for imagenet_c in AS_IMAGENET_IDS_OR_NAMES.get(cname, [])] or None
                bg_paster: PasteOnBackground = PasteOnBackground(
                    background_loader=bg_dataset, bg_type=bg_type,
                    num_imgs=num_bgs_per_ce,
                    forbidden_classes_bg=forbidden_classes, )

                # set the transform (also clears cache if needed)
                vanilla_concept_dataset.transform = bg_paster

            # In any case: make (super) sure the cache is cleared to not leak previous run images:
            if not config.no_in_mem_caching:
                vanilla_concept_dataset.clear_cache()
                assert vanilla_concept_dataset.cache == {}

            ## Iterate over all concept embedding methods
            method_model_combinations = tqdm(list(itertools.product(config.ce_methods, config.model_keys)))
            for count3, (ce_method, model_key) in enumerate(method_model_combinations, 1):
                method_model_combinations.set_description(
                    f"{dataset_key}/{concept_id} "
                    f"({count1}/{len(data_concept_combinations)}), "
                    f"{bg_randomizer_key} w/ #imgs/bg={num_bgs_per_ce} "
                    f"({count2}/{len(randomizer_combinations)}), "
                    f"{ce_method} @ {model_key} "
                    f"({count3}/{len(method_model_combinations)}) "
                    f"{count3 + len(method_model_combinations) * (count2 + len(randomizer_combinations) * count1)}/"
                    f"{len(method_model_combinations) * len(randomizer_combinations) * len(data_concept_combinations)}")

                # Model settings
                model_builder = WRAPPED_MODELS[model_key]
                layers = LAYERS_BY_MODEL[model_key]
                propagator, processor = model_builder(layers, device=config.device)
                activations_extractor = LoCEActivationsTensorExtractor(propagator, model_key, processor=processor)

                curr_spec = f'{dataset_key=}, {concept_id=}, {model_key=}, {bg_randomizer_key=}, {num_bgs_per_ce=}, {ce_method=}'
                common_opt_engine_config = dict(
                    activations_extractor=activations_extractor,
                    datasets=[vanilla_concept_dataset],
                    n_imgs_per_category=config.n_imgs_per_category,
                    target_shape=config.target_shape,
                )
                common_optimizer_config = dict(
                    loce_init='zeros',
                    objective_type=OBJECTIVE_TYPES[ce_method],  # TODO: make enum
                    device=config.device,
                )
                corrected_batch_size = max(1, config.batch_size // num_bgs_per_ce) if config.batch_size is not None else None

                # Do some shortcutting: skip current settings if folder with desired out paths
                exp_settings = dict(run=os.path.basename(config.output_dir),
                                   dataset_key=dataset_key, category_id=concept_id,
                                   bg_randomizer_key=bg_randomizer_key, num_bgs_per_ce=num_bgs_per_ce,
                                   model_key=model_key, ce_method=ce_method,)
                if not config.force_folder_descent and \
                    pkl_todo_count(os.path.dirname(config.output_dir),
                                   **exp_settings,
                                   total_img_counts={dataset_key: len(vanilla_concept_dataset)}) <= 0:
                    continue
                # results_root  /  ce_method/bg_randomizer_key/num_bgs_per_ce/dataset_key
                results_path = os.path.join(os.path.dirname(config.output_dir),
                                            os.path.dirname(os.path.dirname(to_pkl_file_glob(**exp_settings))))
                match ce_method:
                    case "net2vec" | "net2vec_proper_bce":
                        ## Net2Vec:
                        batch_optimizer = TorchCustomLoCEBatchOptimizer(
                            **common_optimizer_config,
                            num_acts_per_loce=None,
                            epochs=1,
                        )
                        optimizer = Net2VecOptimizationEngine(
                            batch_optimizer=batch_optimizer,
                            out_base_dir=results_path,
                            batch_size=config.batch_size,  # note that we don't need the correction here
                            epochs=config.epochs,
                            loading_batch_size=corrected_batch_size,  # for image loading we do
                            **common_opt_engine_config,
                        )
                        log_info(f"STARTING optimization of {curr_spec}  in {results_path}")
                        optimizer.run_optimization(verbose=True, overwrite=True)
                        log_info(f"DONE {curr_spec} in {results_path}")

                        # ## CODE BY GEORGII
                        # # THE CODE BELOW IS DONE ON NORMAL IMAGES!!!!!
                        # coco_segmenter = MSCOCOSemanticSegmentationLoader(coco_annot, selected_category_id)
                        #
                        # activations_extractor = LoCEActivationsTensorExtractor(prop, net_tag, processor=processor)
                        #
                        # # this trains Net2Vec and sparse Net2Vec-16
                        # opt_net2vec = TorchCustomSampleNet2VecOptimizer(activations_extractor, coco_segmenter, [16])
                        #
                        # # vectors, vectors_top16, (images list, segmentation masks), (IoUs, IoUs_top16, Projections, Projections_top16)
                        # net2vec_vectors, net2vec_top16_vectors, (imgs_used, seg_masks), (
                        #     ious_n2v, ious_n2v_top16, proj_n2v, proj_n2v_top16) = opt_net2vec.optimize_net2vec(
                        #     img_names, imgs_path,
                        #     n_samples=n_imgs_per_category)
                        #
                        # write_pickle(
                        #     ((net2vec_vectors, net2vec_top16_vectors), (imgs_used, seg_masks),
                        #      (ious_n2v, ious_n2v_top16), (proj_n2v, proj_n2v_top16)), save_path)

                    case "loce" | "loce_proper_bce":
                        batch_optimizer = TorchCustomLoCEBatchOptimizer(
                            **common_optimizer_config,
                            num_acts_per_loce=num_bgs_per_ce,
                            epochs=config.epochs,
                        )
                        optimizer = LoCEOptimizationEngine(
                            batch_optimizer=batch_optimizer,
                            out_base_dir=results_path,
                            batch_size=corrected_batch_size,
                            **common_opt_engine_config,
                        )
                        log_info(f"STARTING {curr_spec} in {results_path}")
                        optimizer.run_optimization(verbose=True)
                        log_info(f"DONE {curr_spec} in {results_path}")

                    case "netdissect":  # TODO
                        pass
                        # # THE CODE BELOW IS DONE ON NORMAL IMAGES!!!!!
                        # coco_segmenter = MSCOCOSemanticSegmentationLoader(coco_annot, selected_category_id)
                        #
                        # # this trains Net2Vec and sparse Net2Vec-16
                        # # this trains NetDissect, it's represented as sparse vector with only one positive value
                        # opt_netdissect = TorchCustomSampleNetDissectionOptimizer(activations_extractor, coco_segmenter)
                        #
                        # # returns NumPy arrays or Dict[str, np.array], where str is layer name
                        # # vectors Dict[str, np.array], (images list - List[str], segmentation masks - array of arrays), (IoUs - Dict[str, np.array], Projections - Dict[str, np.array])
                        # netdissect_vectors, (imgs_used, seg_masks), (ious_nd, proj_nd) = opt_netdissect.evaluate(img_names,
                        #                                                                                          imgs_path,
                        #                                                                                          n_samples=n_imgs_per_category)
                        #
                        # write_pickle((netdissect_vectors, (imgs_used, seg_masks),
                        #               ious_nd, proj_nd), save_path)

        # finally, clear the cache:
        if not config.no_in_mem_caching:
            vanilla_concept_dataset.clear_cache()
            assert vanilla_concept_dataset.cache == {}

    log_info(f"EXPERIMENT DONE for config {config}")
