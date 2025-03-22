# ========================================================================
# Copyright 2020 Emory University
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
# ========================================================================

# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-08 12:44
import elit
from elit.server import parser_config
from elit.server.en_util import eos, tokenize
from elit.server.service_parser import ServiceParser
from elit.server.service_tokenizer import ServiceTokenizer

service_tokenizer = ServiceTokenizer(eos, tokenize)
service_parser = ServiceParser(
    service_tokenizer=service_tokenizer,
    model=elit.load(parser_config.MTL_MODEL)
)
