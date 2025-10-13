import os
from collections import defaultdict
from typing import TYPE_CHECKING
from typing import Union, Iterable, Dict, Any, Literal, Optional

import torch

from .datasets import SegmentationDataset, ImageLoader
from .semantic_segmenter import MSCOCOSemanticSegmentationLoader, MSCOCORectangleSegmenter, \
    MSCOCOEllipseSegmenter
from ..utils.files import read_json, write_json, mkdir

if TYPE_CHECKING:
    _COCOImgID = int
    _COCOSegmenterTag = Literal['original', 'rectangle', 'ellipse']
    _COCOCatID = int


class MSCOCOAnnotationsProcessor:

    def __init__(self,
                 coco_images_folder: str,
                 annotations_json_path: str,
                 output_path: str = "./data/mscoco2017val/processed/"
                 ) -> None:
        """
        Args:
            coco_images_folder: path to MS COCO images
            annotations_json_path: path to JSON file with MS COCO annotations

        Kwargs:
            output_path: output path
        """
        self.coco_images_folder = coco_images_folder
        self.annotations_json_path = annotations_json_path
        self.output_path = output_path

    def _load_annotations(self) -> Dict[str, Any]:
        """
        Load MS COCO Annotations

        Returns:
            dictionary with annotations
        """
        return read_json(self.annotations_json_path)

    def _save_annotations(self,
                          obj: Dict[str, Any],
                          out_path: str,
                          ) -> None:
        """
        Save processed MS COCO Annotations

        Args:
            obj: dictionary to write as JSON file
            out_path: path to save annotations
        """
        write_json(obj, out_path, "\t")

    def select_relevant_annotations_by_categtory(self,
                                                 categories: Union[int, Iterable[int]],
                                                 json_save_path: str = None
                                                 ) -> Dict[str, Any]:
        """
        Selects only given categories from JSON

        Args:
            categories: MS COCO categories to select
            json_save_path: filename/-path under self.output_path to save annotations;
                set to None to disable saving

        Kwargs:
            json_save_path: save path for the resulting file

        Returns:
            dictionary with MS COCO annotations for given categories
        """
        if isinstance(categories, int):
            categories = [categories]

        annots_json = self._load_annotations()

        # relevant_imgs
        relevant_annots = []
        relevant_img_ids = set()
        for annot in annots_json['annotations']:
            if annot['category_id'] in categories:
                relevant_annots.append(annot)
                relevant_img_ids.add(annot['image_id'])
        annots_json['annotations'] = relevant_annots

        relevant_imgs = []
        for image in annots_json['images']:
            if image['id'] in relevant_img_ids:
                relevant_imgs.append(image)
        annots_json['images'] = relevant_imgs

        if json_save_path is not None:
            mkdir(self.output_path)
            out_path = os.path.join(self.output_path, json_save_path)
            self._save_annotations(annots_json, out_path)

        return annots_json

    def get_person_annotations(self):
        return self.select_relevant_annotations_by_categtory(1, "person_annotations_val2017.json")
    
    def get_person_annotations_va(self):
        pa = self.select_relevant_annotations_by_categtory(1, "person_annotations_val2017.coco.json")
        return self._convert_coco_to_va(pa, "person_annotations_val2017.va.json")

    def _convert_coco_to_va(self, annot: Dict[str, Any], json_save_path: str = None) -> Dict[str, Any]:
        """
        Converts MS COCO to VA format

        Args:
            annot : MS COCO annotations
            json_save_path: save path for the resulting file

        Returns:
            VA annotations
        """
        va_dict_list = {}

        for img in annot['images']:
            img_annots = [a for a in annot['annotations'] if a['image_id'] == img['id']]

            img_path = os.path.join(self.coco_images_folder, img['file_name'])

            va_dict =  {
                "boxes": [],
                "labels": [],
                "boxesVisRatio": [],
                "boxesHeight": [],
                "original_labels": []
            }

            for ia in img_annots:
                va_dict['boxes'].append([int(ia['bbox'][0]), int(ia['bbox'][1]), int(ia['bbox'][0] + ia['bbox'][2]),int(ia['bbox'][1] + ia['bbox'][3])])
                va_dict['labels'].append(ia['category_id'])
                va_dict['boxesVisRatio'].append(1.0)
                va_dict['boxesHeight'].append(int(ia['bbox'][3]))
                va_dict['original_labels'].append(ia['category_id'])

            va_dict_list[img_path] = va_dict

        if json_save_path is not None:
            out_path = os.path.join(self.output_path, json_save_path)
            self._save_annotations(va_dict_list, out_path)

        return va_dict_list


class MSCOCOSegmentationDataset(SegmentationDataset):

    coco_segmenters: dict[Literal['original', 'rectangle', 'ellipse']: 'AbstractSemanticSegmenter'] = {
        'original': MSCOCOSemanticSegmentationLoader,
        'rectangle': MSCOCORectangleSegmenter,
        'ellipse': MSCOCOEllipseSegmenter
    }

    ALL_CAT_NAMES_BY_ID: dict[int, str] = {
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    # These are for MS COCO, but we are using VOC2012
    #     # not used in experiments - commented out
    #     1: 'person',

    #     # vehicles
    #     2: 'bicycle',
    #     3: 'car',
    #     4: 'motorcycle',
    #     5: 'airplane',
    #     6: 'bus',
    #     7: 'train',
    #     8: 'truck',
    #     9: 'boat',

    #     # animals
    #     16: 'bird',
    #     17: 'cat',
    #     18: 'dog',
    #     19: 'horse',
    #     20: 'sheep',
    #     21: 'cow',
    #     22: 'elephant',
    #     23: 'bear',
    #     24: 'zebra',
    #     25: 'giraffe'
    # }

    @property
    def ALL_CAT_NAMES(self) -> list[str]:
        return list(self.ALL_CAT_NAMES_BY_ID.values())

    @property
    def ALL_CAT_IDS(self) -> list[int]:
        return list(self.ALL_CAT_NAMES_BY_ID.keys())

    @property
    def tag(self) -> str:
        prefix = self._tag + "_" if self._tag is not None else ""
        return f"{prefix}{self.segmenter_tag}"

    def __init__(self,
                 imgs_path: str = f"./data/voc2012/VOCdevkit/VOC2012/JPEGImages/",
                 all_annots_path: str =f"./data/voc2012/VOCdevkit/VOC2012/voc2012trainval_annotations.json",
                 processed_annots_dir: str = None,
                 *,
                 segmenter_tag: '_COCOSegmenterTag' = 'original',
                 category_ids: list[int] = None,
                 combine_masks: bool = True,
                 image_loader: ImageLoader = None,
                 transform: Optional['_TransformType'] = None,
                 _annots_by_cat_id: dict[int, dict] = None,
                 device: Union[str, torch.device] = None,
                 **kwargs):
        """

        Args:
            imgs_path: the base folder where to find the images
            all_annots_path: the annotations file containing all annotations (category-specific ones are cached)
            processed_annots_dir: root folder for caching of processed annotations;
                defaults to sibling folder `processed` of imgs_path; set to False to disable caching
            category_ids: the IDs of categories to include
            _annots_by_cat_id: optionally hand over the already processed annotations. Then _load_coco_annot is skipped.
                Useful for crafting own annotations. Make sure to provide a
            **_:
        """
        super().__init__(imgs_path=imgs_path, category_ids=category_ids,
                         combine_masks=combine_masks, image_loader=image_loader,
                         transform=transform,
                         device=device,
                         **kwargs)

        # COCO format specific stuff
        self.segmenter_tag: '_COCOSegmenterTag' = segmenter_tag
        """Segmenter engine used."""

        # set the template for caching annotation files (set to False to deactivate caching)
        if processed_annots_dir is not False:
            processed_annots_dir = processed_annots_dir or os.path.join(os.path.dirname(imgs_path.rstrip(os.path.sep)), "processed")
            self._processed_annots_cache_templ: Union[bool, str] = os.path.join(
                processed_annots_dir,
                os.path.basename(all_annots_path).replace(".json", "_{v_name}.json"))
        else:
            self._processed_annots_cache_templ = False

        self._all_annots_path = all_annots_path
        self._annots_by_cat_id: dict['_COCOCatID', dict[str, Any]] = _annots_by_cat_id if _annots_by_cat_id is not None \
            else self._load_coco_annots(all_annots_path)
        self._img_info_by_img_id: dict['_COCOImgID', dict[str, Any]] = {
            img_info['id']: img_info for annots in self._annots_by_cat_id.values()
            for img_info in annots['images']}
        """Convenience helper to ease accessing image info."""

        self._cat_ids_by_img_id: dict['_COCOImgID', list['_COCOCatID']] = defaultdict(list)
        for cat_id, annots in self._annots_by_cat_id.items():
            for img_info in annots['images']:
                self._cat_ids_by_img_id[img_info['id']].append(cat_id)

        # Filter and populate the img_ids
        self.img_ids: list['_COCOImgID'] = [i for i in self._img_info_by_img_id.keys() if self._check_img_id(i)]
        """The actual image IDs considered by this instance."""
        self._img_info_by_img_id = {i: self._img_info_by_img_id[i] for i in self.img_ids}

    def get_img_filename(self, img_id: '_COCOImgID') -> str:
        img_info = self._img_info_by_img_id[img_id]
        return img_info["file_name"]

    def _load_coco_annots(self, all_annots_path: str = None) -> dict['_COCOCatID', Any]:
        """
        Load and subselect annotations of MSCOCO with MSCOCOAnnotationsProcessor

        Returns:
            annots (Dict[str, Any]): dictionary with only those MSCOCO annotations relevant to self.categories
        """
        all_annots_path = all_annots_path or self._all_annots_path
        annots = {}

        for v_id in self.cat_ids:
            v_name = self.ALL_CAT_NAMES_BY_ID[v_id]

            coco_annot, output_path, json_save_path = None, None, None
            # try reading from cache in case caching is enabled:
            if self._processed_annots_cache_templ is not False:
                annot_path = self._processed_annots_cache_templ.format(v_name=v_name)
                output_path, json_save_path = os.path.dirname(annot_path), os.path.basename(annot_path)
                try:
                    coco_annot = read_json(annot_path)
                except FileNotFoundError:
                    pass

            # in case the file did not exist / caching is disabled:
            if coco_annot is None:
                mcp = MSCOCOAnnotationsProcessor(self.imgs_path, all_annots_path, output_path=output_path)
                coco_annot = mcp.select_relevant_annotations_by_categtory(v_id, json_save_path=json_save_path)

            annots[v_id] = coco_annot

        return annots

    def load_segs(self, img_id: '_COCOImgID') -> dict[int, torch.Tensor]:
        """Load a dictionary {category: segmentation} for the given image."""
        segs: dict[int, torch.Tensor] = {}
        for category_id, annot in self._annots_by_cat_id.items():
            segmenter = self.coco_segmenters[self.segmenter_tag](annot, category_id)
            seg = segmenter.segment_sample(img_id=img_id)
            if seg is not None:
                segs[category_id] = torch.as_tensor(seg, device=self.device)
        return segs

    def get_cats(self, img_id: '_COCOImgID') -> list['_COCOCatID']:
        return self._cat_ids_by_img_id[img_id]

COCO_TO_WORDNET_IDS: dict[str, list[str]] = {
'airplane': ['n02690373'],
 'apple': ['n07742313'],
 'baseball bat': [],
 'baseball glove': [],
 'bear': ['n02132136', 'n02134084', 'n02134418', 'n02133161'],
 'bed': ['n02804414', 'n03125729', 'n03131574', 'n03388549'],
 'bench': ['n03891251'],
 'bicycle': ['n03792782', 'n02835271'],
 'bird': ['n01514668',
          'n01608432',
          'n01860187',
          'n01582220',
          'n02018207',
          'n02006656',
          'n01828970',
          'n02027492',
          'n01833805',
          'n01530575',
          'n01534433',
          'n01531178',
          'n01817953',
          'n01616318',
          'n01558993',
          'n02017213',
          'n01614925',
          'n02018795',
          'n01818515',
          'n01820546',
          'n02037110',
          'n02011460',
          'n01532829',
          'n01560419',
          'n02058221',
          'n02033041',
          'n01843383',
          'n02002556',
          'n02056570',
          'n01622779',
          'n01843065',
          'n01537544',
          'n01824575',
          'n02028035',
          'n01819313',
          'n01592084',
          'n02012849',
          'n02009912',
          'n02002724',
          'n01514859',
          'n02051845',
          'n01855672',
          'n02025239',
          'n02013706',
          'n02007558',
          'n02009229',
          'n01829413',
          'n01580077',
          'n01518878',
          'n01855032',
          'n01601694',
          'n01847000'],
 'boat': ['n03947888',
          'n03095699',
          'n03673027',
          'n04606251',
          'n03662601',
          'n04612504',
          'n04273569',
          'n02951358',
          'n03447447',
          'n03344393'],
 'book': [],
 'bottle': ['n03937543',
            'n03983396',
            'n04560804',
            'n04591713',
            'n04557648',
            'n04579145',
            'n02823428'],
 'bowl': ['n03775546', 'n04263257'],
 'bus': ['n04146614', 'n03769881', 'n04487081'],
 'cake': [],
 'car': ['n03100240',
         'n04285008',
         'n03594945',
         'n04037443',
         'n02701002',
         'n03670208',
         'n02814533',
         'n03777568',
         'n03770679',
         'n02930766'],
 'carrot': [],
 'cat': ['n02128385',
         'n02130308',
         'n02129165',
         'n02128757',
         'n02128925',
         'n02129604'],
 'cell phone': [],
 'chair': ['n04429376', 'n03376595', 'n02791124', 'n04099969'],
 'clock': ['n04548280', 'n02708093', 'n03196217'],
 'couch': ['n04344873'],
 'cow': ['n02408429', 'n02403003', 'n02403003'],
 'dog': ['n02097047',
         'n02090721',
         'n02106550',
         'n02113624',
         'n02085782',
         'n02096585',
         'n02101556',
         'n02095314',
         'n02089078',
         'n02086079',
         'n02090622',
         'n02098413',
         'n02108551',
         'n02088238',
         'n02112018',
         'n02112706',
         'n02112137',
         'n02102318',
         'n02109047',
         'n02110958',
         'n02086240',
         'n02108089',
         'n02093991',
         'n02087394',
         'n02105505',
         'n02108422',
         'n02110185',
         'n02094114',
         'n02102973',
         'n02104365',
         'n02096294',
         'n02100583',
         'n02094258',
         'n02096437',
         'n02099267',
         'n02107312',
         'n02111129',
         'n02111500',
         'n02105162',
         'n02092339',
         'n02096051',
         'n02093428',
         'n02110806',
         'n02102040',
         'n02113978',
         'n02110063',
         'n02090379',
         'n02099601',
         'n02095889',
         'n02106382',
         'n02112350',
         'n02088466',
         'n02093256',
         'n02088364',
         'n02108000',
         'n02091831',
         'n02100877',
         'n02086646',
         'n02101388',
         'n02106662',
         'n02097658',
         'n02110627',
         'n02107574',
         'n02109961',
         'n02105251',
         'n02102480',
         'n02100236',
         'n02097298',
         'n02091635',
         'n02107142',
         'n02099712',
         'n02099849',
         'n02106030',
         'n02085936',
         'n02097474',
         'n02102177',
         'n02093754',
         'n02086910',
         'n02108915',
         'n02085620',
         'n02101006',
         'n02097130',
         'n02105056',
         'n02107908',
         'n02091244',
         'n02105412',
         'n02111277',
         'n02099429',
         'n02113799',
         'n02092002',
         'n02093859',
         'n02113712',
         'n02100735',
         'n02088094',
         'n02093647',
         'n02089973',
         'n02109525',
         'n02105641',
         'n02110341',
         'n02088632',
         'n02113186',
         'n02087046',
         'n02094433',
         'n02095570',
         'n02113023',
         'n02107683',
         'n02098105',
         'n02096177',
         'n02105855',
         'n02097209',
         'n02111889',
         'n02098286',
         'n02106166',
         'n02091467',
         'n02089867',
         'n02104029'],
 'donut': [],
 'elephant': ['n02504013', 'n02504458'],
 'fire hydrant': [],
 'fork': [],
 'frisbee': [],
 'giraffe': [],
 'handbag': [],
 'horse': ['n02389026'],
 'keyboard': ['n03452741', 'n04515003', 'n03854065', 'n04505470'],
 'knife': ['n03041632', 'n03658185'],
 'motorcycle': ['n03785016'],
 'oven': ['n03259280', 'n04111531'],
 'person': ['n09835506', 'n10148035', 'n10565667'],
 'potted plant': [],
 'sandwich': ['n07697313', 'n07697537'],
 'scissors': [],
 'sheep': ['n02412080', 'n02415577'],
 'sink': [],
 'skateboard': [],
 'skis': [],
 'snowboard': [],
 'sports ball': [],
 'stop sign': [],
 'suitcase': [],
 'surfboard': [],
 'tennis racket': [],
 'tie': ['n04591157', 'n02865351', 'n02883205'],
 'toilet': ['n04357314',
            'n03476991',
            'n03314780',
            'n03676483',
            'n03690938',
            'n03916031'],
 'toothbrush': [],
 'train': ['n02917067'],
 'truck': ['n04467665',
           'n03417042',
           'n03977966',
           'n03796401',
           'n03345487',
           'n03930630',
           'n04461696'],
 'tv': [],
 'wine glass': []}


MSCOCO_TO_WORDNET_IDS = {'airplane': ['n02690373'],
 'bear': ['n02133161', 'n02134418', 'n02132136', 'n02134084'],
 'bicycle': ['n03792782', 'n02835271'],
 'bird': ['n01843065',
          'n01817953',
          'n02037110',
          'n02033041',
          'n02007558',
          'n01608432',
          'n01592084',
          'n02017213',
          'n01616318',
          'n01514668',
          'n01614925',
          'n02051845',
          'n02006656',
          'n01828970',
          'n01518878',
          'n01843383',
          'n01531178',
          'n01820546',
          'n01580077',
          'n02011460',
          'n01532829',
          'n02058221',
          'n02012849',
          'n02002724',
          'n02009229',
          'n02028035',
          'n02056570',
          'n02027492',
          'n01534433',
          'n01537544',
          'n02002556',
          'n01601694',
          'n01514859',
          'n01847000',
          'n01622779',
          'n01582220',
          'n02018795',
          'n02013706',
          'n02025239',
          'n01855672',
          'n01560419',
          'n01833805',
          'n02018207',
          'n01530575',
          'n01829413',
          'n01558993',
          'n01818515',
          'n01819313',
          'n01855032',
          'n01860187',
          'n02009912',
          'n01824575'],
 'boat': ['n04612504',
          'n02951358',
          'n03447447',
          'n04273569',
          'n03344393',
          'n03662601',
          'n03947888',
          'n04606251',
          'n03095699',
          'n03673027'],
 'bus': ['n03769881', 'n04487081', 'n04146614'],
 'car': ['n03100240',
         'n04037443',
         'n03594945',
         'n02701002',
         'n03670208',
         'n03770679',
         'n03777568',
         'n02930766',
         'n02814533',
         'n04285008'],
 'cat': ['n02129165',
         'n02129604',
         'n02128385',
         'n02130308',
         'n02128757',
         'n02128925'],
 'cow': ['n02403003', 'n02408429', 'n02403003'],
 'dog': ['n02092339',
         'n02105855',
         'n02097298',
         'n02109961',
         'n02093256',
         'n02105641',
         'n02106030',
         'n02107574',
         'n02108089',
         'n02110958',
         'n02091635',
         'n02093647',
         'n02097658',
         'n02105412',
         'n02112018',
         'n02101556',
         'n02098286',
         'n02107142',
         'n02102040',
         'n02106662',
         'n02098105',
         'n02089867',
         'n02104365',
         'n02111889',
         'n02093754',
         'n02092002',
         'n02087046',
         'n02110627',
         'n02111500',
         'n02099267',
         'n02089078',
         'n02101388',
         'n02106550',
         'n02099601',
         'n02095314',
         'n02085620',
         'n02088466',
         'n02095570',
         'n02101006',
         'n02086079',
         'n02098413',
         'n02097130',
         'n02113186',
         'n02109525',
         'n02108915',
         'n02088238',
         'n02094114',
         'n02090721',
         'n02097047',
         'n02108422',
         'n02085782',
         'n02096051',
         'n02105505',
         'n02104029',
         'n02093428',
         'n02099712',
         'n02096294',
         'n02093991',
         'n02113023',
         'n02088094',
         'n02087394',
         'n02086646',
         'n02090622',
         'n02105056',
         'n02111129',
         'n02102973',
         'n02107683',
         'n02113624',
         'n02091244',
         'n02108551',
         'n02091831',
         'n02096177',
         'n02100877',
         'n02110063',
         'n02110185',
         'n02112350',
         'n02091467',
         'n02108000',
         'n02112137',
         'n02094433',
         'n02110806',
         'n02111277',
         'n02107312',
         'n02109047',
         'n02113712',
         'n02093859',
         'n02096437',
         'n02102177',
         'n02085936',
         'n02112706',
         'n02106166',
         'n02105162',
         'n02106382',
         'n02090379',
         'n02100236',
         'n02100583',
         'n02097474',
         'n02113978',
         'n02086910',
         'n02113799',
         'n02097209',
         'n02099429',
         'n02095889',
         'n02096585',
         'n02105251',
         'n02102318',
         'n02110341',
         'n02099849',
         'n02088364',
         'n02107908',
         'n02102480',
         'n02100735',
         'n02089973',
         'n02086240',
         'n02088632',
         'n02094258'],
 'elephant': ['n02504013', 'n02504458'],
 'giraffe': [],
 'horse': ['n02389026'],
 'motorcycle': ['n03785016'],
 'person': ['n09835506', 'n10565667', 'n10148035'],
 'sheep': ['n02412080', 'n02415577'],
 'train': ['n02917067'],
 'truck': ['n03345487',
           'n04467665',
           'n03796401',
           'n04461696',
           'n03977966',
           'n03930630',
           'n03417042']}


VOC_TO_WORDNET_IDS = {'aeroplane': ['n02690373'],
 'bicycle': ['n02835271', 'n03792782'],
 'bird': ['n02012849',
          'n02013706',
          'n01843383',
          'n01616318',
          'n02009912',
          'n01560419',
          'n02033041',
          'n01828970',
          'n02051845',
          'n02037110',
          'n02056570',
          'n01601694',
          'n01531178',
          'n02028035',
          'n02006656',
          'n01558993',
          'n01580077',
          'n01518878',
          'n01819313',
          'n02002556',
          'n02011460',
          'n01514668',
          'n01622779',
          'n02058221',
          'n02018795',
          'n01530575',
          'n01817953',
          'n01534433',
          'n01582220',
          'n01818515',
          'n02017213',
          'n01847000',
          'n01824575',
          'n02002724',
          'n01860187',
          'n01820546',
          'n02007558',
          'n02018207',
          'n01855032',
          'n01537544',
          'n01833805',
          'n01843065',
          'n01532829',
          'n01829413',
          'n02025239',
          'n02009229',
          'n01514859',
          'n01608432',
          'n01614925',
          'n02027492',
          'n01592084',
          'n01855672'],
 'boat': ['n03673027',
          'n03095699',
          'n04606251',
          'n03947888',
          'n03447447',
          'n04273569',
          'n03662601',
          'n03344393',
          'n02951358',
          'n04612504'],
 'bottle': ['n03983396',
            'n04557648',
            'n02823428',
            'n04579145',
            'n04591713',
            'n04560804',
            'n03937543'],
 'bus': ['n04146614', 'n04487081', 'n03769881'],
 'car': ['n03100240',
         'n03770679',
         'n03594945',
         'n04285008',
         'n03670208',
         'n03777568',
         'n02814533',
         'n02701002',
         'n04037443',
         'n02930766'],
 'cat': ['n02128385',
         'n02128757',
         'n02129604',
         'n02128925',
         'n02130308',
         'n02129165'],
 'chair': ['n02791124', 'n04429376', 'n03376595', 'n04099969'],
 'cow': ['n02403003', 'n02403003', 'n02410509', 'n02408429'],
 'diningtable': ['n03201208'],
 'dog': ['n02105641',
         'n02095314',
         'n02091831',
         'n02107142',
         'n02099267',
         'n02098105',
         'n02109525',
         'n02113712',
         'n02099601',
         'n02100735',
         'n02098286',
         'n02091635',
         'n02105855',
         'n02113023',
         'n02090721',
         'n02102040',
         'n02097047',
         'n02099712',
         'n02089973',
         'n02096585',
         'n02105251',
         'n02110958',
         'n02110627',
         'n02097658',
         'n02096177',
         'n02113978',
         'n02093428',
         'n02093859',
         'n02097474',
         'n02106382',
         'n02109047',
         'n02110806',
         'n02112350',
         'n02091244',
         'n02085620',
         'n02113799',
         'n02102177',
         'n02088364',
         'n02105056',
         'n02097209',
         'n02098413',
         'n02110341',
         'n02106550',
         'n02112137',
         'n02100583',
         'n02113624',
         'n02096294',
         'n02097130',
         'n02087394',
         'n02102973',
         'n02107312',
         'n02101006',
         'n02106030',
         'n02105162',
         'n02092002',
         'n02101388',
         'n02087046',
         'n02099429',
         'n02110063',
         'n02111889',
         'n02088094',
         'n02094433',
         'n02112706',
         'n02094114',
         'n02107574',
         'n02107683',
         'n02086910',
         'n02090379',
         'n02105505',
         'n02106662',
         'n02090622',
         'n02093647',
         'n02102318',
         'n02100877',
         'n02088466',
         'n02093256',
         'n02104365',
         'n02107908',
         'n02096437',
         'n02099849',
         'n02097298',
         'n02096051',
         'n02095889',
         'n02111277',
         'n02102480',
         'n02089867',
         'n02109961',
         'n02101556',
         'n02111500',
         'n02105412',
         'n02089078',
         'n02108000',
         'n02085782',
         'n02108551',
         'n02086240',
         'n02112018',
         'n02091467',
         'n02100236',
         'n02108915',
         'n02104029',
         'n02095570',
         'n02088238',
         'n02086079',
         'n02088632',
         'n02106166',
         'n02086646',
         'n02108089',
         'n02085936',
         'n02093991',
         'n02110185',
         'n02092339',
         'n02093754',
         'n02108422',
         'n02094258',
         'n02113186',
         'n02111129'],
 'horse': ['n02389026'],
 'motorbike': ['n03785016'],
 'person': ['n10148035', 'n09835506', 'n10565667'],
 'pottedplant': ['n12057211', 'n11939491', 'n03991062'],
 'sheep': ['n02412080', 'n02415577'],
 'sofa': ['n04344873'],
 'train': ['n02917067'],
 'tvmonitor': ['n03782006', 'n04404412']}