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
    
    knowledge_base = [
        {"prompt": "Siapa kamu?", "tokens": "Saya adalah MidiLM, AI musik buatan bromanprjkt."},
        {"prompt": "Siapa bromanprjkt?", "tokens": "bromanprjkt adalah pengembang yang menciptakan saya."},
        {"prompt": "Apa itu musik?", "tokens": "Musik adalah seni mengolah suara menjadi harmoni yang indah."},
        {"prompt": "Apa itu piano?", "tokens": "Piano adalah instrumen musik yang dimainkan dengan menekan tuts."},
        {"prompt": "Apa itu gitar?", "tokens": "Gitar adalah instrumen dawai yang dipetik."},
        {"prompt": "Sebutkan genre musik.", "tokens": "Genre musik meliputi Pop, Rock, Jazz, Classical, EDM, dan banyak lagi."},
        {"prompt": "How are you?", "tokens": "I am fine, thank you! How can I help you with music today?"},
        {"prompt": "What is MIDI?", "tokens": "MIDI stands for Musical Instrument Digital Interface, a protocol for digital music."},
    ]
    dataset.extend(knowledge_base)

    general_corpus_id = [
        "Hari ini cuaca sangat cerah sekali.", "Saya suka makan nasi goreng di pagi hari.",
        "Pendidikan adalah kunci masa depan.", "Indonesia adalah negara kepulauan yang indah.",
        "Belajar pemrograman sangatlah menyenangkan.", "Kita harus menjaga lingkungan sekitar.",
        "Terima kasih atas bantuan Anda hari ini.", "Mari kita bekerja sama untuk mencapai tujuan."
    ]
    for text in general_corpus_id:
        dataset.append({"prompt": text[:10], "tokens": text})

    general_corpus_en = [
        "The quick brown fox jumps over the lazy dog.", "Artificial intelligence is changing the world.",
        "I love listening to music while working.", "The sun rises in the east and sets in the west.",
        "Programming is a valuable skill in the modern era.", "We should be kind to everyone we meet.",
        "Thank you for your support and kindness.", "Let's build something amazing together."
    ]
    for text in general_corpus_en:
        dataset.append({"prompt": text[:10], "tokens": text})

    midi_paths = []
    for ext in ["*.mid", "*.midi", "*.MID", "*.MIDI"]:
        midi_paths.extend(list(Path(midi_dir).rglob(ext)))
    
    for midi_path in midi_paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            variations = [(0, 1.0)]
            if augment:
                for ps in range(-1, 2):
                    if ps != 0: variations.append((ps, 1.0))
            
            for ps, ts in variations:
                aug_pm = augment_midi(pm, ps, ts)
                tmp = "tmp_aug.mid"
                aug_pm.write(tmp)
                tokens = midi_to_tokens(tmp)
                if os.path.exists(tmp): os.remove(tmp)
                if not tokens: continue
                
                lang = detect_language(tokens)
                music_str = tokens_to_string(tokens)
                
                if lang == "ID":
                    prompts = [
                        "buatkan saya musik yang enak", "bikin melodi", "saya mau nada buat tidur",
                        "buatkan nada santai", "musik sedih", "melodi bahagia"
                    ]
                else:
                    prompts = [
                        "make me some music", "compose a melody", "I want sleep music",
                        "create chill vibe", "make sad music", "generate happy tune"
                    ]
                
                for p in prompts:
                    dataset.append({"prompt": p, "tokens": music_str})
                    
        except Exception as e:
            print(f"Error {midi_path.name}: {e}")
            
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
