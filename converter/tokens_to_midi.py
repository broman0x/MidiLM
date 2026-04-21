import pretty_midi

def tokens_to_midi(tokens, output_path):
    pm = pretty_midi.PrettyMIDI()
    instruments = {}
    current_time = 0.0
    current_velocity = 64
    current_inst = None
    tempo = 120
    
    for token in tokens:
        if token.startswith("TEMPO_"):
            try:
                tempo = int(token.split("_")[1])
                pm.tick_relative_tempo_changes = [pretty_midi.TempoChange(tempo, 0)]
            except: pass
        elif token.startswith("INST_"):
            name = token
            if name not in instruments:
                if name == "INST_DRUMS":
                    instruments[name] = pretty_midi.Instrument(0, is_drum=True)
                else:
                    prog = int(token.split("_")[1])
                    instruments[name] = pretty_midi.Instrument(prog)
                pm.instruments.append(instruments[name])
            current_inst = instruments[name]
        elif token.startswith("TIME_SHIFT_"):
            try: current_time += int(token.split("_")[2]) / 100.0
            except:
                try: current_time += int(token.split("_")[1]) / 100.0
                except: pass
        elif token.startswith("VELOCITY_"):
            try: current_velocity = (int(token.split("_")[1]) - 1) * 4 + 2
            except: pass
        elif token.startswith("NOTE_ON_"):
            if current_inst is not None:
                try:
                    pitch = int(token.split("_")[2])
                    note = pretty_midi.Note(velocity=current_velocity, pitch=pitch, start=current_time, end=current_time + 0.1)
                    current_inst.notes.append(note)
                except: pass
        elif token.startswith("NOTE_OFF_"):
            if current_inst is not None:
                try:
                    pitch = int(token.split("_")[2])
                    for n in reversed(current_inst.notes):
                        if n.pitch == pitch and n.end == n.start + 0.1:
                            n.end = current_time
                            break
                except: pass
                
    pm.write(output_path)
