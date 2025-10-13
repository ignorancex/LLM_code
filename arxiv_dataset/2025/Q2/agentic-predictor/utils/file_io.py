from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import aiofiles
import aiohttp
import inspect

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from pydantic_core import to_jsonable_python


def handle_unknown_serialization(x: Any) -> str:
    """For `to_jsonable_python` debug, get more detail about the x."""

    if inspect.ismethod(x):
        tip = f"Cannot serialize method '{x.__func__.__name__}' of class '{x.__self__.__class__.__name__}'"
    elif inspect.isfunction(x):
        tip = f"Cannot serialize function '{x.__name__}'"
    elif hasattr(x, "__class__"):
        tip = f"Cannot serialize instance of '{x.__class__.__name__}'"
    elif hasattr(x, "__name__"):
        tip = f"Cannot serialize class or module '{x.__name__}'"
    else:
        tip = f"Cannot serialize object of type '{type(x).__name__}'"

    raise TypeError(tip)


def read_json_file(json_file: str, encoding: str = "utf-8") -> list[Any]:
    if not Path(json_file).exists():
        raise FileNotFoundError(f"json_file: {json_file} not exist, return []")

    with open(json_file, "r", encoding=encoding) as fin:
        try:
            data = json.load(fin)
        except Exception:
            raise ValueError(f"read json file: {json_file} failed")
    return data


def write_json_file(
    json_file: str,
    data: Any,
    encoding: str = "utf-8",
    indent: int = 4,
    use_fallback: bool = False,
):
    folder_path = Path(json_file).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    custom_default = partial(
        to_jsonable_python,
        fallback=handle_unknown_serialization if use_fallback else None,
    )

    with open(json_file, "w", encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=indent, default=custom_default)


def read_jsonl_file(jsonl_file: str, encoding="utf-8") -> list[dict]:
    if not Path(jsonl_file).exists():
        raise FileNotFoundError(f"json_file: {jsonl_file} not exist, return []")
    datas = []
    with open(jsonl_file, "r", encoding=encoding) as fin:
        try:
            for line in fin:
                data = json.loads(line)
                datas.append(data)
        except Exception:
            raise ValueError(f"read jsonl file: {jsonl_file} failed")
    return datas


def add_jsonl_file(jsonl_file: str, data: list[dict], encoding: str = None):
    folder_path = Path(jsonl_file).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    with open(jsonl_file, "a", encoding=encoding) as fout:
        for json_item in data:
            fout.write(json.dumps(json_item) + "\n")


def read_csv_to_list(curr_file: str, header=False, strip_trail=True):
    """
    Reads in a csv file to a list of list. If header is True, it returns a
    tuple with (header row, all rows)
    ARGS:
      curr_file: path to the current csv file.
    RETURNS:
      List of list where the component lists are the rows of the file.
    """
    logger.debug(f"start read csv: {curr_file}")
    analysis_list = []
    with open(curr_file) as f_analysis_file:
        data_reader = csv.reader(f_analysis_file, delimiter=",")
        for count, row in enumerate(data_reader):
            if strip_trail:
                row = [i.strip() for i in row]
            analysis_list += [row]
    if not header:
        return analysis_list
    else:
        return analysis_list[0], analysis_list[1:]


async def aread(filename: str | Path, encoding="utf-8") -> str:
    """Read file asynchronously."""
    if not filename or not Path(filename).exists():
        return ""
    try:
        async with aiofiles.open(str(filename), mode="r", encoding=encoding) as reader:
            content = await reader.read()
    except UnicodeDecodeError:
        async with aiofiles.open(str(filename), mode="rb") as reader:
            raw = await reader.read()
            result = chardet.detect(raw)
            detected_encoding = result["encoding"]
            content = raw.decode(detected_encoding)
    return content


async def awrite(filename: str | Path, data: str, encoding="utf-8"):
    """Write file asynchronously."""
    pathname = Path(filename)
    pathname.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(str(pathname), mode="w", encoding=encoding) as writer:
        await writer.write(data)


async def read_file_block(filename: str | Path, lineno: int, end_lineno: int):
    if not Path(filename).exists():
        return ""
    lines = []
    async with aiofiles.open(str(filename), mode="r") as reader:
        ix = 0
        while ix < end_lineno:
            ix += 1
            line = await reader.readline()
            if ix < lineno:
                continue
            if ix > end_lineno:
                break
            lines.append(line)
    return "".join(lines)


def list_files(root: str | Path) -> List[Path]:
    files = []
    try:
        directory_path = Path(root)
        if not directory_path.exists():
            return []
        for file_path in directory_path.iterdir():
            if file_path.is_file():
                files.append(file_path)
            else:
                subfolder_files = list_files(root=file_path)
                files.extend(subfolder_files)
    except Exception as e:
        logger.error(f"Error: {e}")
    return files


async def aread_bin(filename: str | Path) -> bytes:
    """Read binary file asynchronously.

    Args:
        filename (Union[str, Path]): The name or path of the file to be read.

    Returns:
        bytes: The content of the file as bytes.

    Example:
        >>> content = await aread_bin('example.txt')
        b'This is the content of the file.'

        >>> content = await aread_bin(Path('example.txt'))
        b'This is the content of the file.'
    """
    async with aiofiles.open(str(filename), mode="rb") as reader:
        content = await reader.read()
    return content


async def awrite_bin(filename: str | Path, data: bytes):
    """Write binary file asynchronously.

    Args:
        filename (Union[str, Path]): The name or path of the file to be written.
        data (bytes): The binary data to be written to the file.

    Example:
        >>> await awrite_bin('output.bin', b'This is binary data.')

        >>> await awrite_bin(Path('output.bin'), b'Another set of binary data.')
    """
    pathname = Path(filename)
    pathname.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(str(pathname), mode="wb") as writer:
        await writer.write(data)
