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

# TODO: Randomize on startup and sync with alerting system
KONEY_FINGERPRINT = 1337


def encode_fingerprint_in_echo(code: int) -> str:
    """
    See utils.EncodeFingerprintInEcho in Go code.
    """
    return f"KONEY_FINGERPRINT_{code}"


def encode_fingerprint_in_cat(code: int) -> str:
    """
    See utils.EncodeFingerprintInCat in Go code.
    """
    binary_code = format(code, "b")

    cipher = []
    for bit in binary_code:
        if bit == "0":
            cipher.append("-u")
        else:
            cipher.append("-uu")

    return " ".join(cipher)
