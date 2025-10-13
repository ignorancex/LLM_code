#!/usr/bin/env python3
# coding: utf-8

"""
Generic engine class for cross-modal recommendatino


Author:
- Bereket A. Yilma <name.surname@artaicare.com>
- Luis A. Leiva <name.surname@uni.lu>
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Engine:
    def unpack_prefs(self, preferences):
        """
        Unpack user preferences (music IDs and ratings) into a list of IDs and normalized weights.

        Args:
            preferences (dict): Dictionary of music IDs (str) and ratings (int, 1-5).

        Returns:
            tuple: (music_list, weights) where music_list is a list of music IDs and weights is a list of normalized ratings.
        """
        music_list = []
        weights = []
        for music_id, rating in preferences.items():
            try:
                # Validate rating
                if not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
                    logging.warning(
                        f"Invalid rating {rating} for music ID {music_id}. Rating must be between 1 and 5."
                    )
                    continue
                music_list.append(music_id)
                weights.append(rating)
            except Exception as e:
                logging.error(
                    f"Error processing preference for music ID {music_id}: {str(e)}"
                )
                continue

        if not music_list:
            raise ValueError("No valid preferences provided after validation.")

        xmin = min(weights)
        xmax = max(weights)
        for i, x in enumerate(weights):
            weights[i] = (x - xmin) / (xmax - xmin + sys.float_info.epsilon)

        return music_list, weights
