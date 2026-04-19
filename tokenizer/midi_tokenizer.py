import json
import os
import sys
import pretty_midi

sys.path.insert(0, os.path.dirname(__file__))

SPECIAL_TOKENS = ["PAD", "BOS", "EOS"]
NOTE_ON_TOKENS = [f"NOTE_ON_{p}" for p in range(128)]
NOTE_OFF_TOKENS = [f"NOTE_OFF_{p}" for p in range(128)]
TIME_SHIFT_TOKENS = [f"TIME_SHIFT_{v}" for v in range(1, 101)]
VELOCITY_TOKENS = [f"VELOCITY_{v}" for v in range(1, 33)]
TEMPO_TOKENS = [f"TEMPO_{t}" for t in range(40, 241)]
INST_TOKENS = [f"INST_{i}" for i in range(128)] + ["INST_DRUMS"]

ALL_TOKENS = SPECIAL_TOKENS + NOTE_ON_TOKENS + NOTE_OFF_TOKENS + TIME_SHIFT_TOKENS + VELOCITY_TOKENS + TEMPO_TOKENS + INST_TOKENS
TOKEN_TO_ID = {tok: idx for idx, tok in enumerate(ALL_TOKENS)}
ID_TO_TOKEN = {idx: tok for tok, idx in TOKEN_TO_ID.items()}
VOCAB_SIZE = len(ALL_TOKENS)

PAD_ID = TOKEN_TO_ID["PAD"]
BOS_ID = TOKEN_TO_ID["BOS"]
EOS_ID = TOKEN_TO_ID["EOS"]

def note_name_to_pitch(name):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if "#" in name:
        note_part, octave = name[:2], int(name[2:])
    else:
        note_part, octave = name[0], int(name[1:])
    return (octave + 1) * 12 + notes.index(note_part)

def _velocity_to_bucket(velocity):
    return max(1, min(32, (velocity // 4) + 1))

def _bucket_to_velocity(bucket):
    return min(127, (bucket - 1) * 4 + 2)

def _time_to_shifts(seconds):
    shifts, remaining = [], round(seconds * 100)
    while remaining > 0:
        step = min(remaining, 100)
        shifts.append(f"TIME_SHIFT_{step}")
        remaining -= step
    return shifts

def midi_to_tokens(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    tokens = []
    tempo_changes = pm.get_tempo_changes()
    bpm = int(round(tempo_changes[1][0])) if len(tempo_changes[1]) > 0 else 120
    tokens.append(f"TEMPO_{max(40, min(240, bpm))}")
    events = []
    for inst in pm.instruments:
        name = "INST_DRUMS" if inst.is_drum else f"INST_{inst.program}"
        for note in inst.notes:
            events.append(("on", note.start, note.pitch, note.velocity, name))
            events.append(("off", note.end, note.pitch, 0, name))
    for lyric in pm.lyrics:
        text = lyric.text.strip().replace(" ", "_").upper()
        if text: events.append(("lyric", lyric.time, text, 0, "LYRIC"))
    events.sort(key=lambda e: (e[1], 0 if e[0] == "off" else (1 if e[0] == "lyric" else 2)))
    current_time, current_inst = 0.0, None
    for etype, time, pitch_text, vel, iname in events:
        delta = time - current_time
        if delta > 0.005:
            tokens.extend(_time_to_shifts(delta))
            current_time = time
        if etype == "lyric":
            tokens.append(f"LYRIC_{pitch_text}")
        else:
            if iname != current_inst:
                tokens.append(iname)
                current_inst = iname
            if etype == "on":
                tokens.append(f"VELOCITY_{_velocity_to_bucket(vel)}")
                tokens.append(f"NOTE_ON_{pitch_text}")
            else:
                tokens.append(f"NOTE_OFF_{pitch_text}")
    return tokens

def encode(token_strings, max_len=None):
    global TOKEN_TO_ID, ID_TO_TOKEN, VOCAB_SIZE
    ids = [BOS_ID]
    for tok in token_strings:
        if tok in TOKEN_TO_ID: ids.append(TOKEN_TO_ID[tok])
        elif tok.startswith("LYRIC_"):
            if tok not in TOKEN_TO_ID:
                TOKEN_TO_ID[tok] = VOCAB_SIZE
                ID_TO_TOKEN[VOCAB_SIZE] = tok
                VOCAB_SIZE += 1
            ids.append(TOKEN_TO_ID[tok])
    ids.append(EOS_ID)
    if max_len:
        ids = ids[:max_len]
        while len(ids) < max_len: ids.append(PAD_ID)
    return ids

def decode(token_ids):
    tokens = []
    for tid in token_ids:
        tok = ID_TO_TOKEN.get(tid)
        if tok and tok not in SPECIAL_TOKENS: tokens.append(tok)
    return tokens

def tokens_from_string(s):
    global TOKEN_TO_ID, ID_TO_TOKEN, VOCAB_SIZE
    res = []
    for tok in s.strip().split():
        if tok.startswith("LYRIC_"):
            res.append(tok)
            if tok not in TOKEN_TO_ID:
                TOKEN_TO_ID[tok] = VOCAB_SIZE
                ID_TO_TOKEN[VOCAB_SIZE] = tok
                VOCAB_SIZE += 1
        elif tok.startswith("NOTE_ON_") and not tok[8:].isdigit():
            try: res.append(f"NOTE_ON_{note_name_to_pitch(tok[8:])}")
            except: res.append(tok)
        elif tok.startswith("NOTE_OFF_") and not tok[9:].isdigit():
            try: res.append(f"NOTE_OFF_{note_name_to_pitch(tok[9:])}")
            except: res.append(tok)
        else: res.append(tok)
    return res

def tokens_to_string(l):
    return " ".join(l)
