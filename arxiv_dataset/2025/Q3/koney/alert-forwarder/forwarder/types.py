# Copyright (c) 2025 Dynatrace LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import Literal, TypedDict


class ContainerMetadata(TypedDict):
    id: str
    name: str


class PodMetadata(TypedDict):
    name: str
    namespace: str
    container: ContainerMetadata


class NodeMetadata(TypedDict):
    name: str


class ProcessMetadata(TypedDict):
    uid: int
    pid: int
    cwd: str
    binary: str
    arguments: str


class KoneyAlert(TypedDict):
    timestamp: str  # ISO 8601
    deception_policy_name: str | None
    trap_type: Literal[
        "unknown",
        "filesystem_honeytoken",
        "http_endpoint",
        "http_payload",
    ]

    # optional metadata that can be present depending on the trap type
    metadata: dict
    pod: PodMetadata | None
    node: NodeMetadata | None
    process: ProcessMetadata | None


DynatraceSeverity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]


class DynatraceSink(TypedDict):
    api_url: str
    api_token: str
    # TODO (#27): specify severity in the policy, not in the sink!
    severity: DynatraceSeverity


class AlertSink(TypedDict):
    name: str
    dynatrace_sink: DynatraceSink | None
