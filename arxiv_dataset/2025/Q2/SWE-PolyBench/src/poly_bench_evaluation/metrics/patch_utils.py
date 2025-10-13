# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, Union

from loguru import logger
from tree_sitter import Node
from unidiff import PatchSet
from whatthepatch import parse_patch

from poly_bench_evaluation.metrics.tree_sitter_utils import (
    EXTENSION_LANGUAGE_MAP,
    TREE_SITTER_FUNC_CLASS_TYPES,
    TREE_SITTER_TOP_LEVEL_NODE_TYPES,
)
from poly_bench_evaluation.repo_utils import RepoManager

from .node_utils import (
    _find_identifiable_parent,
    _get_node_identifier,
    get_node_by_line_number,
    get_parent_id,
)


class Patch:
    """Class that provides helper functions to interact with patches."""

    def __init__(self, patch: str):
        # Parse patch
        self.patch_str = patch
        self.patch = list(parse_patch(patch))
        self.patch_set = None

        try:
            self.patch_set = PatchSet(patch)
        except Exception as e:
            logger.warning(
                f"Could not parse patch as PatchSet. Won't be able to compute node retrieval metrics. {e}"
            )

        # Read target files
        self.files: List[Path] = []
        self._get_files()

    def _get_files(self):
        """Fills the list of files that were modified in the patch."""
        for diff in self.patch:
            file_path = diff.header.old_path
            if file_path.startswith("a/"):
                file_path = file_path[2:]
            self.files.append(Path(file_path))

    def get_modified_files(self, to_basename: bool = False) -> Set[str]:
        """Returns a set of files that are part of the patch.

        Args:
            to_basename: If true, return the basename of the files instead of the full path.

        Yields:
            A generator containing the stringified file paths or basenames touched in the patch.
        """
        patch_files = []
        for file in self.files:
            if to_basename:
                patch_files.append(file.name)
            else:
                patch_files.append(str(file))

        return set(patch_files)

    def _add_modified_line(
        self,
        line_collections: List[Union[Set[str], Dict[str, Set[int]]]],
        file_name: str,
        line_num: str,
        as_dict: bool,
    ) -> None:
        """
        Helper function to add modified lines to collections.

        Args:
            line_collections: List of collections to add the line to
            file_name: Name of the file being modified
            line_num: Line number as string
            as_dict: Whether to store as dictionary or set
        """
        for collection in line_collections:
            if as_dict:
                assert isinstance(collection, defaultdict)
                collection[file_name].add(int(line_num))
            else:
                assert isinstance(collection, set)
                collection.add(f"{file_name}:{line_num}")

    def _get_modified_lines_by_status(self, as_dict: bool = True) -> Tuple[
        Union[Set[str], Dict[str, Set[int]]],
        Union[Set[str], Dict[str, Set[int]]],
        Union[Set[str], Dict[str, Set[int]]],
    ]:
        """
        Get the modified lines in the patch. Line numbers refer to old/new file.

        """
        modified_old_lines: Union[Set[str], Dict[str, Set[int]]] = (
            set() if not as_dict else defaultdict(set)
        )
        modified_new_lines: Union[Set[str], Dict[str, Set[int]]] = (
            set() if not as_dict else defaultdict(set)
        )

        modified_lines: Union[Set[str], Dict[str, Set[int]]] = (
            set() if not as_dict else defaultdict(set)
        )

        assert self.patch_set is not None, "patch was not parsed to PatchSet"
        for patched_file in self.patch_set:
            # Extract file names
            old_file_name = patched_file.source_file
            new_file_name = patched_file.target_file

            if old_file_name.startswith("a/"):
                old_file_name = old_file_name[2:].strip()
            if new_file_name.startswith("b/"):
                new_file_name = new_file_name[2:].strip()

            if old_file_name == new_file_name:
                for hunk in patched_file:
                    for line in hunk:
                        if line.is_removed:
                            self._add_modified_line(
                                [modified_old_lines, modified_lines],
                                old_file_name,
                                str(line.source_line_no),
                                as_dict,
                            )
                        if line.is_added:
                            self._add_modified_line(
                                [modified_new_lines, modified_lines],
                                new_file_name,
                                str(line.target_line_no),
                                as_dict,
                            )
        return modified_old_lines, modified_new_lines, modified_lines

    def get_modified_lines(self, as_dict: bool = False) -> Union[Set[str], Dict[str, Set[int]]]:
        """
        Get the modified lines in the patch.

        Args:
            as_dict: If true, return a dictionary where the keys are file names
                     and the values are lists of line numbers.

        Returns:
            If as_dict is False, a list of strings containing an entry for each modified line
            represented by the file name and the line number of the modified line.
            If as_dict is True, a dictionary where thge keys are file names
            and the value are lists of line numbers.
        """
        modified_lines: Union[Set[str], Dict[str, Set[int]]] = (
            set() if not as_dict else defaultdict(set)
        )

        for diff in self.patch:
            if diff.changes:
                file_name = diff.header.old_path
                if file_name.startswith("a/"):
                    file_name = file_name[2:]

                file_name = file_name.strip()
                for old_line, new_line, _ in diff.changes:
                    if old_line is None or new_line is None:
                        # If old_line is None,  the line was added
                        # If new_line is None, the line was removed
                        line_number = old_line or new_line
                        if as_dict:
                            assert isinstance(modified_lines, defaultdict)
                            modified_lines[file_name].add(int(line_number))
                        else:
                            assert isinstance(modified_lines, set)
                            modified_lines.add(f"{file_name}:{line_number}")
        return modified_lines

    def _get_nodes(self, modified_lines, repo_root_path):
        def _get_node_key(node):
            return (node.start_point, node.end_point)

        def get_all_parent_ids(
            node: Node, current_list: List[str], visited: Optional[Set[str]] = None
        ) -> List[str]:
            if visited is None:
                visited = set()

            # Get immediate parent
            parent_id = get_parent_id(node)
            if parent_id is None or parent_id in visited:
                return current_list

            visited.add(parent_id)
            current_list.append(parent_id)

            # Get the parent node from _find_identifiable_parent
            parent_node = _find_identifiable_parent(node)

            # Continue recursion with the found parent
            if parent_node is not None:
                return get_all_parent_ids(parent_node, current_list, visited)

            return current_list

        file_nodes: DefaultDict[str, Set[Node]] = defaultdict(set)
        for file, lines in modified_lines.items():

            extension = Path(file).suffix
            language_name = EXTENSION_LANGUAGE_MAP.get(extension)
            if not language_name:
                logger.info(f"Can't build CST for {extension} file. Skipping.")
                continue

            if len(lines) > 100:
                logger.warning(
                    f"File {file} has {len(lines)} modified lines. This may take a while."
                )

            file_path = Path(repo_root_path) / file
            node_set = [get_node_by_line_number(file_path, line) for line in lines]

            node_set = [
               node
               for node in node_set
               if node is not None and node.type not in TREE_SITTER_TOP_LEVEL_NODE_TYPES
            ]

            _visited = set()
            unique_nodes = [
               node
               for node in node_set
               if not (_get_node_key(node) in _visited or _visited.add(_get_node_key(node)))  # type: ignore[func-returns-value]
            ]

            for node in unique_nodes:
                if node is not None:
                    if (
                        node.type not in TREE_SITTER_TOP_LEVEL_NODE_TYPES
                        and node.type in TREE_SITTER_FUNC_CLASS_TYPES
                    ):
                        # Get function or class name
                        func_or_class_name = _get_node_identifier(node)

                        # Get parent's identifier
                        all_nodes = get_all_parent_ids(node, [])[::-1]
                        all_nodes.append(f"{node.type}:{func_or_class_name}")

                        node_str = "->".join(all_nodes)
                        file_nodes[file].add(node_str)

        return file_nodes

    def get_modified_nodes(self, repo_manager: RepoManager) -> DefaultDict[str, Set[Node]]:
        """Get the modified nodes in the patch. Only if a line contained in a function of class definition,
        will the node be returned. For example, definition of variables outside a function or class will be ignored.

        Args:
            repo_root_path: The root path of the repository.

        Returns:
            A dictionary where the keys are file names and the values are sets of nodes
            representing the modified nodes in the patch.
        """
        assert (
            self.patch_set is not None
        ), "patch was not parsed to PatchSet during __init__. Can't compute node retrieval metrics."

        old_modified_lines, new_modified_lines, _ = self._get_modified_lines_by_status(as_dict=True)

        assert isinstance(old_modified_lines, defaultdict)

        # Get nodes at the current state of the repository
        repo_root_path = repo_manager.tmp_repo_dir

        logger.info("Getting nodes before applying patch")
        pre_change_nodes = self._get_nodes(old_modified_lines, repo_root_path)

        logger.info("applying patch")
        repo_manager.apply_patch(self.patch_str)

        logger.info("Getting nodes after applying patch")
        post_change_nodes = self._get_nodes(new_modified_lines, repo_root_path)

        merged = defaultdict(set)
        for d in [pre_change_nodes, post_change_nodes]:
            for k, v in d.items():
                merged[k].update(v)

        return merged
