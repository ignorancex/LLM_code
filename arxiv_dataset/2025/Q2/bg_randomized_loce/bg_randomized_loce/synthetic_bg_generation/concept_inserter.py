import os
import random
from typing import List, Any, Dict

import numpy as np
from PIL import Image

from .constants import BG_ARTIFICIAL, BG_DIR
from ..data_structures.semantic_segmenter import AbstractSemanticSegmenter


class ConceptInserter:
    """
    Inserts a given MS COCO concept (category_id) onto various backgrounds
    by picking the i-th background for the i-th image (index-based).
    Center-crops each (1024x1024) background to match the MS COCO image size.
    """

    def __init__(
        self,
        cat_id: int,
        cat_name: str,
        segmenter: AbstractSemanticSegmenter,
        coco_annot: Dict[str, Any],
        images_path: str,
        background_base_dir: str = BG_DIR,
        background_types: List[str] = BG_ARTIFICIAL,
        min_coverage: float = 0.0,
        max_coverage: float = 1.0,
        seed_foreground: int = 42,
        seed_background: int = 42,
    ):
        """
        Args:
            cat_id (int): The MS COCO category ID (e.g., 17 for "cat").
            cat_name (str): The MS COCO category name (e.g., "cat" for 17).
            segmenter (AbstractSemanticSegmenter): The semantic segmenter for the category.
            coco_annot (Dict[str, Any]): COCO annotation JSON.
            images_path (str): Directory where MS COCO images are located.
            background_base_dir (str): Base directory with subfolders of backgrounds.
            background_types (List[str]): List of background subfolder names.
            min_coverage (float): Minimum proportion of the image that must be covered by the mask (0-1).
            max_coverage (float): Maximum proportion of the image that can be covered by the mask (0-1).
        """
        self.cat_id = cat_id
        self.cat_name = cat_name
        self.images_path = images_path
        self.background_base_dir = background_base_dir

        self.background_types = background_types
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage

        # Load COCO annotation JSON content
        self.coco_annot = coco_annot

        # Create segmenter
        self.segmenter = segmenter

        # Preload background files for efficiency
        self.background_files = {
            bg_type: sorted(os.listdir(os.path.join(self.background_base_dir, bg_type)))
            for bg_type in self.background_types
        }

        # Set seeds for reproducibility
        self.random_fg = random.Random(seed_foreground)
        self.random_bg = random.Random(seed_background)

    def run(self, num_samples: int, output_dir: str):
        """
        Main entry point:
        1) Filter images containing the desired category.
        2) Compute segmentation and filter by mask coverage.
        3) Sort filtered images by descending mask coverage.
        4) Select the top `num_samples` images.
        5) Shuffle the selected images.
        6) Generate synthetic images.

        Args:
            num_samples (int): Number of images to sample.
            seed (int): Seed for random sampling.
            output_dir (str): Output directory for the generated images.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Filter images with the desired category
        cat_annots = [a for a in self.coco_annot["annotations"] if a["category_id"] == self.cat_id]
        image_ids_with_cat = set(a["image_id"] for a in cat_annots)
        all_coco_images = self.coco_annot["images"]
        filtered_images = [img_info for img_info in all_coco_images if img_info["id"] in image_ids_with_cat]

        valid_images = []

        # Compute mask coverage and filter by range
        for img_info in filtered_images:
            img_name = img_info["file_name"]

            # Generate binary segmentation mask
            seg_mask = self.segmenter.segment_sample(img_name)

            # Compute mask coverage
            mask_coverage = np.sum(seg_mask) / seg_mask.size

            # Only keep images within the valid mask coverage range
            if self.min_coverage <= mask_coverage <= self.max_coverage:
                valid_images.append((img_info, mask_coverage))

        # Sort by mask coverage (descending)
        valid_images.sort(key=lambda x: x[1], reverse=True)

        # Select top `num_samples` images
        selected_images = valid_images[:min(num_samples, len(valid_images))]

        # Shuffle the selected images
        self.random_fg.shuffle(selected_images)

        counter = 0

        # Process each selected image
        for img_idx, (img_info, mask_coverage) in enumerate(selected_images):
            img_name = img_info["file_name"]
            img_name_no_ext = os.path.splitext(img_name)[0]
            original_img_path = os.path.join(self.images_path, img_name)

            # Generate binary segmentation mask again
            seg_mask = self.segmenter.segment_sample(img_name)

            # Extract object as RGBA
            object_img_rgba = self._extract_object(original_img_path, seg_mask)

            # Save original background image
            out_filename = f"{img_name_no_ext}_{self.cat_name}_original.jpg"
            out_path = os.path.join(output_dir, out_filename)
            
            pil_image = Image.open(original_img_path).convert("RGB")
            pil_image.save(out_path, format="JPEG")

            # Process each background type
            for bg_type in self.background_types:
                bg_files = self.background_files[bg_type]
                if not bg_files:
                    print(f"Warning: Background folder '{bg_type}' is empty. Skipping.")
                    continue

                # Select a background using the BG seed
                chosen_bg = self.random_bg.choice(bg_files)
                chosen_bg_path = os.path.join(self.background_base_dir, bg_type, chosen_bg)
                bg_base = os.path.splitext(os.path.basename(chosen_bg_path))[0]

                # Overlay object on chosen background
                synthetic_img = self._overlay_object_on_background(chosen_bg_path, object_img_rgba)

                # Save synthetic image
                out_filename = f"{img_name_no_ext}_{self.cat_name}_{bg_base}.jpg"
                out_path = os.path.join(output_dir, out_filename)

                synthetic_img_rgb = synthetic_img.convert("RGB")
                synthetic_img_rgb.save(out_path, format="JPEG")
                counter += 1

        print(f"Done generating {counter} synthetic '{self.cat_name}' images for.")

    @staticmethod
    def _extract_object(original_img_path: str, binary_mask: np.ndarray) -> Image.Image:
        """
        Extracts the object as RGBA using the given binary mask (True where object is).
        """
        pil_image = Image.open(original_img_path).convert("RGB")

        mask_uint8 = (binary_mask.astype(np.uint8)) * 255
        pil_mask = Image.fromarray(mask_uint8, mode='L')

        pil_image_rgba = pil_image.convert("RGBA")
        object_img_rgba = Image.new("RGBA", pil_image_rgba.size, (0, 0, 0, 0))
        object_img_rgba.paste(pil_image_rgba, (0, 0), mask=pil_mask)
        return object_img_rgba

    @staticmethod
    def _overlay_object_on_background(background_path: str, object_img_rgba: Image.Image) -> Image.Image:
        """
        Load the background (1024x1024).
        Center-crop it to the size of object_img_rgba.
        Paste the object at (0,0).
        Return the composited image.
        """
        target_width, target_height = object_img_rgba.size

        bg_img = Image.open(background_path).convert("RGBA")
        bg_w, bg_h = bg_img.size

        # Center-crop
        left = (bg_w - target_width) // 2
        top = (bg_h - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        cropped_bg = bg_img.crop((left, top, right, bottom))

        # Paste the object
        cropped_bg.paste(object_img_rgba, (0, 0), object_img_rgba)
        return cropped_bg
    
    def evaluate_coverage(self, min_coverage: float, max_coverage: float):
        """
        Evaluate segmentation mask coverage for images with a specific category.

        Args:
            min_coverage (float): Minimum acceptable mask coverage.
            max_coverage (float): Maximum acceptable mask coverage.

        Outputs:
            - Number of images within the specified mask coverage range.
            - Minimum and maximum segmentation mask coverage found across all images.
        """
        # Filter images with the desired category
        cat_annots = [a for a in self.coco_annot["annotations"] if a["category_id"] == self.cat_id]
        image_ids_with_cat = set(a["image_id"] for a in cat_annots)
        all_coco_images = self.coco_annot["images"]
        filtered_images = [img_info for img_info in all_coco_images if img_info["id"] in image_ids_with_cat]

        # Initialize counters and min/max coverage tracking
        counter = 0
        min_coverage_found = float("inf")
        max_coverage_found = float("-inf")

        for img_idx, img_info in enumerate(filtered_images):
            img_name = img_info["file_name"]
            original_img_path = os.path.join(self.images_path, img_name)

            # Generate binary segmentation mask
            seg_mask = self.segmenter.segment_sample(img_name)

            # Compute mask coverage
            mask_coverage = np.sum(seg_mask) / seg_mask.size

            # Update min/max coverage
            min_coverage_found = min(min_coverage_found, mask_coverage)
            max_coverage_found = max(max_coverage_found, mask_coverage)

            # Count images within the specified range
            if min_coverage <= mask_coverage <= max_coverage:
                counter += 1

        # Print results
        print(f"{self.cat_id}".rjust(2) + " " + f"{self.cat_name}".ljust(15) + f": " +
            f"{counter}".ljust(3) + " of " + f"{len(filtered_images)}".ljust(3) + 
            f" images have mask coverage within the range {min_coverage:.0%}-{max_coverage:.0%} (min/max overall: {min_coverage_found:.2%}/{max_coverage_found:.2%}).")