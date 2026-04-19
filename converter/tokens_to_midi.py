import os
import sys
import pretty_midi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer.midi_tokenizer import _bucket_to_velocity

def tokens_to_midi(tokens, output_path="outputs/output.mid"):
    tempo = 120
    for tok in tokens:
        if tok.startswith("TEMPO_"):
            try:
                tempo = int(tok.split("_")[1])
                tempo = max(40, min(240, tempo))
            except: pass
            break
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instruments = {}
    current_time = 0.0
    current_velocity = 100
    current_inst_name = "INST_0"
    active_notes = {}
    for tok in tokens:
        if tok.startswith("TEMPO_"): continue
        elif tok.startswith("TIME_SHIFT_"):
            try: current_time += int(tok.split("_")[2]) / 100.0
            except: pass
        elif tok.startswith("LYRIC_"):
            text = tok[6:].replace("_", " ")
            pm.lyrics.append(pretty_midi.Lyric(text=text, time=current_time))
        elif tok.startswith("VELOCITY_"):
            try: current_velocity = _bucket_to_velocity(int(tok.split("_")[1]))
            except: pass
        elif tok.startswith("INST_"):
            current_inst_name = tok
        elif tok.startswith("NOTE_ON_"):
            try:
                pitch = int(tok.split("_")[2])
                if current_inst_name not in active_notes: active_notes[current_inst_name] = {}
                active_notes[current_inst_name][pitch] = (current_time, current_velocity)
            except: pass
        elif tok.startswith("NOTE_OFF_"):
            try:
                pitch = int(tok.split("_")[2])
                if current_inst_name in active_notes and pitch in active_notes[current_inst_name]:
                    start, velocity = active_notes[current_inst_name].pop(pitch)
                    end = max(current_time, start + 0.05)
                    if current_inst_name not in instruments:
                        is_drum = current_inst_name == "INST_DRUMS"
                        prog = 0 if is_drum else int(current_inst_name.split("_")[1])
                        instruments[current_inst_name] = pretty_midi.Instrument(program=prog, is_drum=is_drum)
                    instruments[current_inst_name].notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
            except: pass
    for iname, notes in active_notes.items():
        for pitch, (start, vel) in notes.items():
            if iname not in instruments:
                is_d = iname == "INST_DRUMS"
                p = 0 if is_d else int(iname.split("_")[1])
                instruments[iname] = pretty_midi.Instrument(program=p, is_drum=is_d)
            instruments[iname].notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=current_time + 0.25))
    for inst in instruments.values(): pm.instruments.append(inst)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pm.write(output_path)
    return output_path
