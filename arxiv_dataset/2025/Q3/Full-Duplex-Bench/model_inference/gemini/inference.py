#!/usr/bin/env python3
"""
Function:
1. Load input.wav and convert it to 16 kHz mono PCM-16.
2. Split it into 30 ms chunks (no local VAD), stream to Gemini for server-side VAD segmentation and response.
3. After each Gemini response completes (generation_complete or turn_complete) or times out, end the current session and start a new one with remaining chunks.
4. Write all Gemini TTS responses (24 kHz) to output.wav, matching the duration of input.wav. Maintain silence where no TTS is generated.
5. Ensure the program exits properly without hanging due to no response.
"""
import os
import asyncio
import time
import math
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import pyaudio

from google import genai
from google.genai import types
from glob import glob
from tqdm import tqdm

task = "user_interruption"

overwrite = True
root_dir = f"/home/daniel094144/data-full-duplex-bench/v1/{task}/"
root_file_dir = f"{root_dir}/*/input.wav"

MODEL = "gemini-2.0-flash-live-001"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"


wav_files = sorted(glob(root_file_dir))

SEND_SR = 16_000
RECV_SR = 24_000

FRAME_MS = 30
FRAME_SAMPLES = SEND_SR * FRAME_MS // 1000  # 480 samples per chunk

REC_TICK_MS = 20
CHANNELS, FORMAT = 1, pyaudio.paInt16

CONFIG = {
    "response_modalities": ["AUDIO"],
    "speech_config": {
        "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}
    },
    "system_instruction": "You are a helpful assistant.",
    "realtime_input_config": {"automatic_activity_detection": {"disabled": False}},
}
# ─────────────────────────────────────────────────


def resample_to_16k(input_path: Path) -> Tuple[Path, float]:
    """
    Resample input.wav to 16kHz mono PCM. Return new path and original duration.
    """
    data, sr = sf.read(input_path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    duration = len(data) / sr

    import scipy.signal as ss

    g = math.gcd(sr, SEND_SR)
    data_16k = ss.resample_poly(data, SEND_SR // g, sr // g)
    data_16k = (data_16k / np.max(np.abs(data_16k)) * 32767).astype(np.int16)

    out_path = input_path.with_name(f"{input_path.stem}_16k_mono.wav")
    sf.write(out_path, data_16k, SEND_SR, subtype="PCM_16")
    print(
        f"[DEBUG] Resampling complete → {out_path} (Original duration: {duration:.2f}s)"
    )
    return out_path, duration


def frame_iter(wav16k_path: Path) -> List[bytes]:
    """
    Read 16kHz mono PCM-16 wav and split into 30 ms chunks.
    """
    data, _ = sf.read(wav16k_path, dtype="int16")
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.int16)

    pad = (-len(data)) % FRAME_SAMPLES
    if pad:
        data = np.pad(data, (0, pad))

    total_frames = len(data) // FRAME_SAMPLES
    frames: List[bytes] = []
    for i in range(total_frames):
        chunk = data[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        frames.append(chunk.tobytes())

    return frames


class Recorder:
    """
    Responsible for playing Gemini's audio and writing it to output.wav.
    """

    def __init__(self, out_sr: int, target_sec: float, outfile="output.wav"):
        self.out_sr = out_sr
        self.target_samples = int(round(target_sec * out_sr))
        self.tick_samples = out_sr * REC_TICK_MS // 1000
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.count = 0
        self.muted = False

        print(f"[DEBUG] Recorder initialized, target samples = {self.target_samples}")
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=out_sr, output=True
        )
        self.wav = sf.SoundFile(
            outfile, "w", samplerate=out_sr, channels=1, subtype="PCM_16"
        )
        self._silence = np.zeros(self.tick_samples, np.int16)

    async def add(self, pcm: bytes):
        if self.muted:
            print("[DEBUG] Received new TTS. Unmuting and resuming playback.")
            self.muted = False
        await self.queue.put(pcm)

    def interrupt(self):
        print(
            "[DEBUG] Recorder: Interrupted received. Clearing queue and entering muted state."
        )
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.muted = True

    async def run(self):
        print("[DEBUG] Recorder: Playback coroutine started")
        while self.count < self.target_samples:
            if not self.muted and not self.queue.empty():
                pcm = await self.queue.get()
                smp = np.frombuffer(pcm, np.int16)
            else:
                remain = self.target_samples - self.count
                n = min(self.tick_samples, remain)
                smp = self._silence[:n]

            try:
                await asyncio.to_thread(self.stream.write, smp.tobytes())
            except OSError as e:
                print("⚠️ Recorder write error:", e)
            self.wav.write(smp)
            self.count += len(smp)

            if self.muted and self.queue.empty():
                await asyncio.sleep(self.tick_samples / self.out_sr)

        print("[DEBUG] Recorder: Playback and writing finished")

        self.wav.close()
        self.stream.close()
        self.p.terminate()


async def end_audio_turn(sess):
    try:
        print("[DEBUG] send audio_stream_end")
        await sess.send_realtime_input(audio_stream_end=True)
    except TypeError:
        print(
            "[DEBUG] audio_stream_end not supported, fallback to end_realtime_input()"
        )
        try:
            await sess.end_realtime_input()
        except AttributeError:
            print("[WARN] end_realtime_input() not supported, skipping")


async def run_session(
    client: genai.Client,
    session_id: int,
    frames: List[bytes],
    start_idx: int,
    rec: Recorder,
) -> int:
    print(
        f"[DEBUG][Session {session_id}] Starting new Gemini Session from idx {start_idx}"
    )
    session_done = False
    idx = start_idx
    total = len(frames)
    tts_buffers: List[bytes] = []

    async with client.aio.live.connect(model=MODEL, config=CONFIG) as sess:

        async def sender():
            nonlocal idx, session_done
            while idx < total and not session_done:
                buf = frames[idx]
                await sess.send_realtime_input(
                    audio=types.Blob(data=buf, mime_type="audio/pcm;rate=16000")
                )
                idx += 1
                await asyncio.sleep(FRAME_MS / 1000)

            if not session_done:
                print(
                    f"[DEBUG][Session {session_id}] All chunks sent. Sending audio_stream_end."
                )
                await end_audio_turn(sess)

        async def receiver():
            nonlocal session_done
            blocked = False
            async for resp in sess.receive():
                sc = resp.server_content

                if getattr(sc, "interrupted", False):
                    print(
                        f"[DEBUG][Session {session_id}] Interrupted received → rec.interrupt() and audio_stream_end"
                    )
                    rec.interrupt()
                    await end_audio_turn(sess)
                    session_done = True
                    return

                if blocked and (
                    getattr(sc, "turn_complete", False)
                    or getattr(sc, "generation_complete", False)
                ):
                    print(
                        f"[DEBUG][Session {session_id}] turn_complete/generation_complete received (blocked), ending round"
                    )
                    session_done = True
                    return

                if sc and sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data:
                            pcm = part.inline_data.data
                            tts_buffers.append(pcm)
                            await rec.add(pcm)

                        if getattr(part, "generation_complete", False):
                            print(
                                f"[DEBUG][Session {session_id}] part.generation_complete → ending round"
                            )
                            session_done = True
                            return

                if getattr(sc, "turn_complete", False) or getattr(
                    sc, "generation_complete", False
                ):
                    print(
                        f"[DEBUG][Session {session_id}] turn_complete/generation_complete → ending round"
                    )
                    session_done = True
                    return

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())

        await sender_task

        try:
            await asyncio.wait_for(receiver_task, timeout=5.0)
        except asyncio.TimeoutError:
            print(f"[DEBUG][Session {session_id}] Receiver timeout. Ending session.")
            receiver_task.cancel()
        except Exception:
            receiver_task.cancel()
            raise

    print(
        f"[DEBUG][Session {session_id}] Gemini Session closed. Next chunk idx = {idx}\n"
    )
    return idx


async def main(input_wav, output_wav):
    input_path = Path(input_wav)
    if not input_path.exists():
        print(f"[ERROR] {input_wav} not found.")
        return False

    if os.path.exists(output_wav) and not overwrite:
        print(f"[WARN] {output_wav} already exists. Skipping.")
        return True

    wav16k_path, orig_dur = resample_to_16k(input_path)

    frames = frame_iter(wav16k_path)
    total_chunks = len(frames)
    print(f"[DEBUG] Total chunks = {total_chunks} (30 ms/each)")

    total_input_samples = int(orig_dur * RECV_SR)
    rec = Recorder(RECV_SR, orig_dur, outfile=output_wav)
    rec_task = asyncio.create_task(rec.run())

    client = genai.Client(api_key=GEMINI_API_KEY)

    chunk_idx = 0
    session_id = 1
    while chunk_idx < total_chunks:
        print(
            f"=== Session {session_id} started (from chunk {chunk_idx}, total {total_chunks}) ==="
        )
        try:
            new_idx = await run_session(client, session_id, frames, chunk_idx, rec)
        except Exception:
            print(f"[ERROR][Session {session_id}] Exception occurred. Exiting loop.")
            traceback.print_exc()
            break

        if new_idx == chunk_idx:
            print(
                f"[WARN] Session {session_id} did not advance chunk_idx. Forcing exit."
            )
            break

        chunk_idx = new_idx
        session_id += 1

    await rec_task
    print("[DEBUG] All sessions complete. Program ended normally.")
    return True


if __name__ == "__main__":
    try:
        for wav_file in tqdm(wav_files):
            print(f"Processing: {wav_file}")
            input_wav = wav_file
            output_wav = wav_file.replace("input.wav", "output.wav")
            while True:
                try:
                    result = asyncio.run(main(input_wav, output_wav))
                    if result is True:
                        print("[INFO] main() executed successfully. Stopping retries.")
                        break
                    else:
                        print("[WARN] main() failed. Retrying soon...")
                except Exception as e:
                    print(f"[ERROR] Exception occurred: {e}. Retrying soon...")
                time.sleep(1)
    except KeyboardInterrupt:
        print("⇪ Program interrupted by user")
    except Exception:
        traceback.print_exc()
