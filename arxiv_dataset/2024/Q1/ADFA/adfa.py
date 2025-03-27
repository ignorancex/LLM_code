import numpy as np
import librosa

def dft(n, scale=None):
    """
    Modify from scipy.linalg.dft implementation of SciPy.

    Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    omegas = np.exp(-2j * np.pi * np.arange(n) / n).reshape(-1, 1)
    m = omegas ** np.arange(n)
    if scale == 'sqrtn':
        m /= math.sqrt(n)
    elif scale == 'n':
        m /= n
    return m

def adfa_arb(n, n_arb):
    """
    Modify from scipy.linalg.dft implementation of SciPy.
    n_arb is the lenght of the signal you want to analyze.
    """
    omegas = np.exp(-1j * np.pi * np.arange(n) / (n - 1)).reshape(-1, 1)
    m = omegas ** np.arange(n_arb)
    return m

def mdfa_arb(n, fs, n_arb):
    """
    Modify from scipy.linalg.dft implementation of SciPy.
    n_arb is the lenght of the signal you want to analyze.
    """
    mel_frequencies=librosa.mel_frequencies(n_mels=n, fmin=0.0, fmax=fs/2, htk=False)
    omegas = np.exp(-1j * np.pi * mel_frequencies / (fs/2)).reshape(-1, 1)
    m = omegas ** np.arange(n_arb)
    return m

def cqfa_arb(n, base, bins, n_arb):
    """
    Modify from scipy.linalg.dft implementation of SciPy.
    n_arb is the lenght of the signal you want to analyze.
    When base=2, bins means the number of frequency components per octave.
    """
    omegas = np.exp(-1j * np.pi * pow(base, -np.arange(n)/bins)[::-1]).reshape(-1, 1)
    m = omegas ** np.arange(n_arb)
    return m

def cqa_arb(n, base, bins, n_arb):
    """
    Modify from scipy.linalg.dft implementation of SciPy.
    n_arb is the lenght of the signal you want to analyze.
    When base=2, bins means the number of frequency components per octave.
    """
    # omegas = np.exp(-1j * np.pi * pow(pow(base, -1/int(bins)), np.arange(n))[::-1]).reshape(-1, 1)
    omegas = np.exp(-1j * np.pi * pow(base, -np.arange(n)/int(bins))[::-1]).reshape(-1, 1)
    m = omegas ** np.arange(n_arb)
    return m
