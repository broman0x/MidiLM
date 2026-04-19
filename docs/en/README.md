# MidiLM Documentation (English)

MidiLM is a specialized Transformer model designed for music and singing generation.

## How it works
1. **Audio Transcription**: You provide MP3 files. The system uses AI to extract MIDI notes.
2. **Tokenization**: Music and Lyrics are converted into text-like "tokens".
3. **Training**: The model learns to predict the next token in a sequence (e.g., if it sees a "DO" note followed by "RE", it learns the pattern).
4. **Inference**: You give it a prompt like [LANG: EN] Happy song, and it generates a new sequence of musical and lyric tokens.
5. **Reconstruction**: The tokens are turned back into a MIDI file, which can be played as an MP3 using a SoundFont.

## Requirements
- Python 3.12
- PyTorch (for the AI Brain)
- TensorFlow (for the MP3-to-MIDI transcription)
- FluidSynth (to hear the results)
