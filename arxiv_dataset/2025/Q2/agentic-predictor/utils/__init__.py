from __future__ import annotations

from PIL import Image
from urllib.parse import quote, unquote
from serpapi import GoogleSearch
from openai import OpenAI
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from pathlib import Path
from abc import ABC, abstractmethod

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_transformers import LongContextReorder

import json
import requests
import base64
import io
import re
import pymupdf
import numpy as np
import aiofiles
import aiohttp


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def print_message(sender, msg, agent_name="Agent", pid=None):
    pid = f"-{pid}" if pid else ""
    sender_color = {
        "user": color.PURPLE,
        "system": color.RED,
        "manager": color.GREEN,
        "agent": color.BLUE,
        "log": color.DARKCYAN,
    }
    sender_label = {
        "user": "💬 You:",
        "system": "⚠️ SYSTEM NOTICE ⚠️\n",
        "log": "📋 LOG:",
        "manager": "🕴🏻 Agent Manager:",
        "agent": "👾 " + agent_name + ":",
    }

    msg = f"{color.BOLD}{sender_color[sender]}{sender_label[sender]}{color.END}{color.END} {msg} \r\n"
    print(msg)



def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)