#!/usr/bin/env python3
"""Push-to-talk voice dictation: hold key → record → release → transcribe → clipboard."""

import argparse
import select
import subprocess
import sys
import time

import numpy as np

import evdev
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

# evdev key codes
KEY_MAP = {
    "f9": evdev.ecodes.KEY_F9,
    "f10": evdev.ecodes.KEY_F10,
    "f12": evdev.ecodes.KEY_F12,
    "scrolllock": evdev.ecodes.KEY_SCROLLLOCK,
    "pause": evdev.ecodes.KEY_PAUSE,
}


def find_keyboards():
    devices = []
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        caps = dev.capabilities()
        key_caps = caps.get(evdev.ecodes.EV_KEY, [])
        if len(key_caps) > 50:
            devices.append(dev)
    return devices


def to_clipboard(text):
    subprocess.run(["wl-copy", "--", text], check=True)


def beep(freq=800, duration=0.08):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * freq * t)
    sd.play(tone, SAMPLE_RATE, blocking=False)


def main():
    parser = argparse.ArgumentParser(description="Push-to-talk voice dictation")
    parser.add_argument(
        "--key", default="scrolllock", choices=list(KEY_MAP.keys()),
        help="Push-to-talk key (default: scrolllock)",
    )
    parser.add_argument(
        "--model", default="small", help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--language", default="es", help="Language code (default: es)",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Audio input device index",
    )
    args = parser.parse_args()

    target_key = KEY_MAP[args.key]

    keyboards = find_keyboards()
    if not keyboards:
        print("No keyboards found. Are you in the 'input' group?", file=sys.stderr)
        print("Run: sudo usermod -aG input $USER && reboot", file=sys.stderr)
        sys.exit(1)

    kb_names = ", ".join(kb.name for kb in keyboards)
    print(f"Keyboards: {kb_names}")

    print(f"Loading whisper model '{args.model}'...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print("Model loaded.")

    print(f"Push-to-talk key: {args.key.upper()}")
    print("Hold to record, release to transcribe. Ctrl+C to quit.\n")

    recording = False
    audio_chunks = []

    def record_callback(indata, frames, time_info, status):
        if recording:
            audio_chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=args.device,
        callback=record_callback,
    )
    stream.start()

    def refresh_keyboards():
        for kb in keyboards:
            try:
                kb.close()
            except Exception:
                pass
        keyboards.clear()
        keyboards.extend(find_keyboards())
        if keyboards:
            names = ", ".join(kb.name for kb in keyboards)
            print(f"  Keyboards reconnected: {names}")
        return len(keyboards) > 0

    try:
        while True:
            if not keyboards:
                time.sleep(1)
                if refresh_keyboards():
                    continue
                continue

            try:
                r, _, _ = select.select(keyboards, [], [])
            except (OSError, ValueError):
                print("\n  Device lost during select, reconnecting...", flush=True)
                recording = False
                refresh_keyboards()
                continue

            for kb in r:
                try:
                    events = list(kb.read())
                except OSError:
                    print("\n  Device lost during read, reconnecting...", flush=True)
                    recording = False
                    refresh_keyboards()
                    break

                for event in events:
                    if event.type != evdev.ecodes.EV_KEY or event.code != target_key:
                        continue

                    if event.value == 1 and not recording:
                        audio_chunks.clear()
                        recording = True
                        beep(600, 0.06)
                        print("● Recording...", end="", flush=True)

                    elif event.value == 0 and recording:
                        recording = False
                        beep(900, 0.06)
                        print(" done.", flush=True)

                        if not audio_chunks:
                            print("  (no audio captured)")
                            continue

                        audio = np.concatenate(audio_chunks)
                        duration = len(audio) / SAMPLE_RATE
                        if duration < 0.3:
                            print(f"  (too short: {duration:.1f}s)")
                            continue

                        print(f"  Transcribing {duration:.1f}s of audio...")
                        audio_f32 = audio.flatten().astype(np.float32) / 32768.0

                        segments, info = model.transcribe(
                            audio_f32,
                            language=args.language,
                            beam_size=5,
                            vad_filter=True,
                        )
                        text = " ".join(seg.text.strip() for seg in segments).strip()

                        if text:
                            to_clipboard(text)
                            print(f"  → {text}")
                            print("  (copied to clipboard)")
                        else:
                            print("  (no speech detected)")

    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        stream.stop()
        stream.close()
        for kb in keyboards:
            try:
                kb.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
