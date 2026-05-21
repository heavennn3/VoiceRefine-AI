import os
import ollama
from faster_whisper import WhisperModel

def run_zero_cost_pipeline(audio_file_path):
    # --- STAGE 1: LOCAL AUDIO TRANSCRIPTION ---
    print("🔄 Stage 1: Initializing local audio engine...")
    
    # "base" or "small" are perfect for standard laptops. They download completely free.
    # 'compute_type="int8"' compresses the model math so your CPU/GPU runs it fast.
    audio_model = WhisperModel("small", device="cpu", compute_type="int8")
    
    print(f"🎙️ Transcribing audio file: {audio_file_path}...")
    segments, info = audio_model.transcribe(audio_file_path, beam_size=5)
    
    # Merge speech segments together into one raw transcript
    raw_transcript = " ".join([segment.text for segment in segments])
    
    print("\n--- RAW TRANSCRIPT GENERATED ---")
    print(raw_transcript)
    print("--------------------------------\n")
    
    # --- STAGE 2: LOCAL TEXT REFINEMENT ---
    print("🧠 Stage 2: Passing raw text to local LLM for word removal...")
    
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
            model='qwen2.5-coder:7b',
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': raw_transcript}
            ]
        )
        
        polished_transcript = response['message']['content']
        return raw_transcript, polished_transcript

    except Exception as e:
        return raw_transcript, f"Error calling local LLM: {str(e)}. Ensure Ollama is running."

# --- TEST THE SCRIPT ---
if __name__ == "__main__":
    # Place a sample recording file in the same directory to test it out
    sample_file = "test_speech.wav" 
    
    if os.path.exists(sample_file):
        raw, clean = run_zero_cost_pipeline(sample_file)
        print("✨ FINAL POLISHED TRANSCRIPT: ✨")
        print(clean)
    else:
        print(f"⚠️ Please put a real audio file named '{sample_file}' here to test the zero-cost script.")