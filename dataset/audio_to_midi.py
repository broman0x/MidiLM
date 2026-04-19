import os
import argparse
import sys
import basic_pitch
import librosa
from pathlib import Path
from basic_pitch.inference import predict_and_save

def transcribe_audio(audio_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    audio_extensions = [".mp3", ".wav", ".ogg", ".flac", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(Path(audio_dir).rglob(f"*{ext}")))
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    print(f"Found {len(audio_files)} audio files. Starting transcription...")
    for audio_path in audio_files:
        expected_midi = Path(output_dir) / f"{audio_path.stem}_basic_pitch.mid"
        if expected_midi.exists():
            print(f"Skipping (already exists): {audio_path.name}")
            continue
        
        try:
            duration = librosa.get_duration(path=str(audio_path))
            if duration > 600:
                print(f"Skipping (too long: {duration/60:.1f} min): {audio_path.name}")
                continue
        except: pass

        print(f"Processing: {audio_path.name}...")
        try:
            predict_and_save(
                audio_path_list=[str(audio_path)],
                output_directory=output_dir,
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=str(basic_pitch.ICASSP_2022_MODEL_PATH)
            )
        except Exception as e:
            print(f"Failed to process {audio_path.name}: {e}")
            continue
    print(f"Transcription complete. MIDI files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="midis/transcribed")
    args = parser.parse_args()
    transcribe_audio(args.audio_dir, args.output_dir)
