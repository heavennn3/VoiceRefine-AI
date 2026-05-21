import os
import ollama
from faster_whisper import WhisperModel


def voice_pipeline(audio_file_path):
    print("Stage 1 : Initialize local audio engine")
    audio_model = WhisperModel("small", device="cpu", compute_type="int8")
    print(f"Transcribing audio file: {audio_file_path}...")

    segments, info = audio_model.transcribe(audio_file_path, beam_size=5)
    raw_transcript = " ".join([segment.text for segment in segments])

    print("\n ---ORIGINAL TRANSCRIPT---")
    print(raw_transcript)
    print("\n ---ORIGINAL TRANSCRIPT---")

    print("Stage 2 : Pass raw text to LLM ")

    system_instruction = (
        "You are an AI audio transcript optimizer. Your task is to clean verbal tics. "
        "1. Remove filler words entirely: 'uh', 'um', 'like', 'you know', 'ah', 'so'. "
        "2. Remove accidental word stutters or exact phrase repetitions. "
        "3. Fix punctuation and paragraph spacing, but keep the person's exact vocabulary and ideas. "
        "Do NOT summarize, do NOT rewrite into formal essays, and do NOT add explanatory text. "
        "Output ONLY the final cleaned transcript."
    )

    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': raw_transcript}
            ]
        )
        refined_transcript = response['message']['content']
        return raw_transcript, refined_transcript
    except Exception as e:
        return raw_transcript, "ERROR calling local LLM"
        

    
    


