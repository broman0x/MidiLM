import os
import subprocess
import sys

SOUNDFONT_PATHS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/default.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    "/usr/share/soundfonts/FluidR3_GM2-2.sf2",
]

def find_soundfont():
    for path in SOUNDFONT_PATHS:
        if os.path.isfile(path): return path
    for search_dir in ["/usr/share/sounds", "/usr/share/soundfonts"]:
        if os.path.isdir(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith(".sf2"): return os.path.join(root, f)
    return None

def midi_to_wav(midi_path, wav_path, soundfont_path=None):
    if soundfont_path is None: soundfont_path = find_soundfont()
    if soundfont_path is None: raise FileNotFoundError("No SoundFont found.")
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    cmd = ["fluidsynth", "-ni", soundfont_path, midi_path, "-F", wav_path, "-r", "44100"]
    subprocess.run(cmd, capture_output=True, text=True)
    return wav_path

def wav_to_mp3(wav_path, mp3_path):
    os.makedirs(os.path.dirname(mp3_path) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "192k", mp3_path]
    subprocess.run(cmd, capture_output=True, text=True)
    return mp3_path

def midi_to_mp3(midi_path, mp3_path, soundfont_path=None, keep_wav=False):
    wav_path = mp3_path.rsplit(".", 1)[0] + ".wav"
    midi_to_wav(midi_path, wav_path, soundfont_path)
    wav_to_mp3(wav_path, mp3_path)
    if not keep_wav and os.path.isfile(wav_path): os.remove(wav_path)
    return mp3_path
