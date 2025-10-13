# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download datasets and specific captures from the datasets."""

from __future__ import annotations

from typing import Union

import tyro
from typing_extensions import Annotated

from scrstudio.scripts.downloads.aachen import AachenDownload
from scrstudio.scripts.downloads.naver_lab import NaverDownload
from scrstudio.scripts.downloads.utils import DatasetDownload

Commands = Union[
    Annotated[NaverDownload, tyro.conf.subcommand(name="naver")],
    Annotated[AachenDownload, tyro.conf.subcommand(name="aachen")],
]



def main(
    dataset: DatasetDownload,
):
    """Script to download existing datasets.
    We currently support the datasets listed above in the Commands.

    Args:
        dataset: The dataset to download (from).
    """
    dataset.save_dir.mkdir(parents=True, exist_ok=True)

    dataset.download(dataset.save_dir)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
