import json
import os
import argparse
import sys
import copy
from pathlib import Path
import pretty_midi

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tokenizer"))

from midi_tokenizer import midi_to_tokens, tokens_to_string

ID_WORDS = {"DAN", "YANG", "DI", "KE", "DENGAN", "INI", "ITU", "SAYA", "KITA", "ADA", "BISA", "TIDAK", "AKU", "KAMU"}
EN_WORDS = {"THE", "AND", "TO", "OF", "A", "IN", "IS", "THAT", "IT", "YOU", "FOR", "WITH", "WAS"}

def detect_language(tokens):
    words = [t[6:] for t in tokens if t.startswith("LYRIC_")]
    if not words: return "INST"
    id_count = sum(1 for w in words if w in ID_WORDS)
    en_count = sum(1 for w in words if w in EN_WORDS)
    if id_count > en_count: return "ID"
    if en_count > id_count: return "EN"
    return "MULTI"

def augment_midi(pm, ps=0, ts=1.0):
    new_pm = copy.deepcopy(pm)
    for inst in new_pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                note.pitch = max(0, min(127, note.pitch + ps))
        if ts != 1.0:
            for note in inst.notes:
                note.start *= ts
                note.end *= ts
    for lyric in new_pm.lyrics:
        lyric.time *= ts
    return new_pm

def prepare_dataset(midi_dir, output_file, augment=False):
    dataset = []
    midi_paths = []
    for ext in ["*.mid", "*.midi", "*.MID", "*.MIDI"]:
        midi_paths.extend(list(Path(midi_dir).rglob(ext)))
    print(f"Found {len(midi_paths)} MIDI files.")
    for midi_path in midi_paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            variations = [(0, 1.0)]
            if augment:
                for ps in range(-3, 4):
                    if ps != 0: variations.append((ps, 1.0))
                variations.append((0, 0.95))
                variations.append((0, 1.05))
            for ps, ts in variations:
                aug_pm = augment_midi(pm, ps, ts)
                tmp = "tmp_aug.mid"
                aug_pm.write(tmp)
                tokens = midi_to_tokens(tmp)
                if os.path.exists(tmp): os.remove(tmp)
                if not tokens: continue
                lang = detect_language(tokens)
                tempo = 120
                for t in tokens:
                    if t.startswith("TEMPO_"):
                        tempo = t.split("_")[1]
                        break
                mood = "VOCAL" if any(t.startswith("LYRIC_") for t in tokens) else "UNKNOWN"
                prompt = f"[LANG: {lang}] [TEMPO: {tempo}] [MOOD: {mood}] MIDI {midi_path.name}"
                dataset.append({"prompt": prompt, "tokens": tokens_to_string(tokens)})
                print(f"Processed: {midi_path.name} ({lang})")
        except Exception as e:
            print(f"Error {midi_path.name}: {e}")
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved: {output_file} ({len(dataset)} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="dataset/example_dataset.json")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    prepare_dataset(args.midi_dir, args.output_file, args.augment)
