#!/usr/bin/env python3
"""
Gradio Web UI for Qwen3-TTS

Supports multiple Qwen3-TTS models with text-to-speech capabilities:
- CustomVoice: Premium voice profiles with intelligent voice control
- VoiceDesign: Create custom voices using natural language descriptions
- Base (Voice Clone): Clone voices from reference audio clips

Model variants: 12Hz (1.7B and 0.6B)
Languages: 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
"""

import os
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# Base directory for models
MODELS_BASE_DIR = os.environ.get("MODELS_BASE_DIR", "/models")

# Available models mapping
ALL_MODELS = {
    "Qwen3-TTS-12Hz-1.7B-CustomVoice": f"{MODELS_BASE_DIR}/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen3-TTS-12Hz-0.6B-CustomVoice": f"{MODELS_BASE_DIR}/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": f"{MODELS_BASE_DIR}/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen3-TTS-12Hz-0.6B-VoiceDesign": f"{MODELS_BASE_DIR}/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
    "Qwen3-TTS-12Hz-1.7B-Base": f"{MODELS_BASE_DIR}/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen3-TTS-12Hz-0.6B-Base": f"{MODELS_BASE_DIR}/Qwen3-TTS-12Hz-0.6B-Base",
}


def check_model_exists(model_path):
    """Check if model exists on disk."""
    config_path = os.path.join(model_path, "config.json")
    return os.path.exists(config_path)


def get_available_models():
    """Filter models to only include those that exist on disk."""
    available = {}
    for name, path in ALL_MODELS.items():
        if check_model_exists(path):
            available[name] = path
            print(f"✓ Found model: {name} at {path}")
        else:
            print(f"✗ Model not found: {name} at {path}")
    return available


AVAILABLE_MODELS = get_available_models()


def should_enable_generate(selected_model_name):
    """Check if generate button should be enabled based on loaded model."""
    return model is not None and selected_model_name == loaded_model_name


def get_available_devices():
    """Get list of available compute devices with descriptive names."""
    devices = {}

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            display_name = f"CUDA{i}: {device_name}"
            devices[display_name] = f"cuda:{i}"
            print(f"Found CUDA device {i}: {device_name}")

    cpu_info = ""

    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_info = line.split(":", 1)[1].strip()
                    break
    except:
        pass

    if cpu_info:
        display_name = f"CPU0: {cpu_info}"
    else:
        display_name = "CPU0"
    devices[display_name] = "cpu"

    return devices


def load_model_with_device(model_name, device_display):
    """Load the Qwen3-TTS model on specified device."""
    global \
        model, \
        supported_speakers, \
        supported_languages, \
        speaker_display_map, \
        lang_display_map, \
        loaded_model_name

    model_path = AVAILABLE_MODELS.get(model_name)
    device_map = available_devices[device_display]

    print(f"Loading model from: {model_path}")
    print(f"Using device: {device_display} ({device_map})")

    use_cuda = device_map.startswith("cuda")

    try:
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            attn_implementation="flash_attention_2" if use_cuda else "eager",
        )

        supported_speakers = model.get_supported_speakers()
        supported_languages = model.get_supported_languages()

        speaker_display_map = {s.title(): s for s in supported_speakers}
        lang_display_map = {l.title(): l for l in supported_languages}

        loaded_model_name = model_name
        return f"✓ {model_name} loaded successfully on {device_display}!", True
    except Exception as e:
        return f"✗ Error loading {model_name}: {str(e)}", False


def unload_model():
    """Unload the current model and free memory."""
    global model, loaded_model_name

    if model is not None:
        print("Unloading current model...")
        del model
        model = None
        loaded_model_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")

        import gc

        gc.collect()
        print("Model unloaded successfully")


# Global variables for model and metadata
model = None
supported_speakers = []
supported_languages = []
speaker_display_map = {}
lang_display_map = {}
available_devices = {}
selected_model_type = None
loaded_model_name = None


def generate_speech(text, language, speaker, instruct, ref_audio=None, ref_text=None):
    """
    Generate speech using CustomVoice, VoiceDesign, or Base model.

    Args:
        text: Input text to synthesize
        language: Target language (display name)
        speaker: Speaker voice profile (display name, for CustomVoice only)
        instruct: Optional instruction for tone/emotion control
        ref_audio: Reference audio file for voice cloning (Base model only)
        ref_text: Transcript of reference audio (Base model only)

    Returns:
        Tuple of (sample_rate, audio_data) for Gradio audio output
    """
    global selected_model_type

    if model is None:
        return None, "✗ Error: Model not loaded. Please load the model first."

    if not text.strip():
        return None, "Error: Please enter text to synthesize."

    try:
        # Auto-detect language if "Auto" is selected
        if language == "Auto":
            language = None
        else:
            # Map display name to model name
            language = lang_display_map.get(language, language)

        # Choose generation method based on model type
        if selected_model_type == "VoiceDesign":
            # VoiceDesign model - doesn't use speaker parameter
            wavs, sr = model.generate_voice_design(
                text=text.strip(),
                language=language,
                instruct=instruct.strip() if instruct else None,
            )
            speaker_info = "Voice Design (custom voice description)"
        elif selected_model_type == "Base":
            # Base model - for voice cloning
            if not ref_audio:
                return (
                    None,
                    "Error: Voice Clone model requires reference audio. Please upload a reference audio file.",
                )
            if not ref_text or not ref_text.strip():
                return (
                    None,
                    "Error: Voice Clone model requires reference text transcript. Please provide the transcript of the reference audio.",
                )

            ref_audio_path = ref_audio if isinstance(ref_audio, str) else None
            wavs, sr = model.generate_voice_clone(
                text=text.strip(),
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text.strip(),
            )
            speaker_info = f"Voice Clone (reference: {ref_audio_path})"
        else:
            # CustomVoice model - requires speaker parameter
            speaker = speaker_display_map.get(speaker, speaker)
            wavs, sr = model.generate_custom_voice(
                text=text.strip(),
                language=language,
                speaker=speaker,
                instruct=instruct.strip() if instruct else None,
            )
            speaker_info = f"Speaker: {speaker}"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wavs[0], sr)
            output_path = f.name

        # Convert float32 audio to int16 for Gradio
        audio_int16 = (wavs[0] * 32767).astype(np.int16)

        return (
            (sr, audio_int16),
            f"✓ Generated successfully! {speaker_info}, Language: {language or 'Auto'}",
        )

    except Exception as e:
        return None, f"✗ Error: {str(e)}"


def create_demo():
    """Create the Gradio demo interface."""
    global available_devices, selected_model_type

    # Determine default model type for initial visibility
    default_model = list(AVAILABLE_MODELS.keys())[0] if AVAILABLE_MODELS else ""
    if "VoiceDesign" in default_model:
        selected_model_type = "VoiceDesign"
    elif "Base" in default_model:
        selected_model_type = "Base"
    else:
        selected_model_type = "CustomVoice"
    is_voice_design_visible = selected_model_type == "VoiceDesign"
    is_custom_voice_visible = selected_model_type == "CustomVoice"
    is_voice_clone_visible = selected_model_type == "Base"
    print(f"[create_demo] Default model: {default_model}, Type: {selected_model_type}")
    print(f"[create_demo] Initial voice_design_tips visible: {is_voice_design_visible}")
    print(f"[create_demo] Initial custom_voice_tips visible: {is_custom_voice_visible}")
    print(f"[create_demo] Initial voice_clone_tips visible: {is_voice_clone_visible}")

    # Build speaker description dictionary (lowercase keys to match model output)
    speaker_descriptions = {
        "vivian": "Bright, slightly edgy young female voice. (Native: Chinese)",
        "serena": "Warm, gentle young female voice. (Native: Chinese)",
        "uncle_fu": "Seasoned male voice with a low, mellow timbre. (Native: Chinese)",
        "dylan": "Youthful Beijing male voice with a clear, natural timbre. (Native: Chinese, Beijing Dialect)",
        "eric": "Lively Chengdu male voice with a slightly husky brightness. (Native: Chinese, Sichuan Dialect)",
        "ryan": "Dynamic male voice with strong rhythmic drive. (Native: English)",
        "aiden": "Sunny American male voice with a clear midrange. (Native: English)",
        "ono_anna": "Playful Japanese female voice with a light, nimble timbre. (Native: Japanese)",
        "sohee": "Warm Korean female voice with rich emotion. (Native: Korean)",
    }

    def update_speaker_info(speaker):
        """Update speaker description when speaker is selected."""
        return speaker_descriptions.get(speaker_display_map.get(speaker, speaker), "")

    def update_model_info(model_name):
        """Update UI based on selected model."""
        global selected_model_type
        if "VoiceDesign" in model_name:
            selected_model_type = "VoiceDesign"
        elif "Base" in model_name:
            selected_model_type = "Base"
        else:
            selected_model_type = "CustomVoice"
        print(f"[update_model_info] Model: {model_name}, Type: {selected_model_type}")
        print(
            f"[update_model_info] voice_design_tips visible: {selected_model_type == 'VoiceDesign'}"
        )
        print(
            f"[update_model_info] custom_voice_tips visible: {selected_model_type == 'CustomVoice'}"
        )
        print(
            f"[update_model_info] voice_clone_tips visible: {selected_model_type == 'Base'}"
        )
        return (
            gr.update(
                visible=(selected_model_type == "VoiceDesign")
            ),  # voice_design_tips
            gr.update(
                visible=(selected_model_type == "CustomVoice")
            ),  # custom_voice_tips
            gr.update(visible=(selected_model_type == "Base")),  # voice_clone_tips
            gr.update(visible=(selected_model_type == "Base")),  # ref_audio_input
            gr.update(visible=(selected_model_type == "Base")),  # ref_text_input
            gr.update(interactive=should_enable_generate(model_name)),  # generate_btn
        )

    def init_tips():
        """Initialize tips visibility based on default model selection."""
        global selected_model_type
        default_model = list(AVAILABLE_MODELS.keys())[0] if AVAILABLE_MODELS else ""
        if "VoiceDesign" in default_model:
            selected_model_type = "VoiceDesign"
        elif "Base" in default_model:
            selected_model_type = "Base"
        else:
            selected_model_type = "CustomVoice"
        print(f"[init_tips] Model: {default_model}, Type: {selected_model_type}")
        print(
            f"[init_tips] voice_design_tips visible: {selected_model_type == 'VoiceDesign'}"
        )
        print(
            f"[init_tips] custom_voice_tips visible: {selected_model_type == 'CustomVoice'}"
        )
        print(f"[init_tips] voice_clone_tips visible: {selected_model_type == 'Base'}")
        return (
            gr.update(visible=selected_model_type == "VoiceDesign"),
            gr.update(visible=selected_model_type == "CustomVoice"),
            gr.update(visible=selected_model_type == "Base"),
        )

    def load_model_handler(model_name, device):
        """Handle model loading and update UI."""
        unload_model()
        status, success = load_model_with_device(model_name, device)

        if success:
            speaker_choices = list(speaker_display_map.keys())
            lang_choices = ["Auto"] + [
                l for l in lang_display_map.keys() if l != "Auto"
            ]

            print(
                f"[load_model_handler] Model: {model_name}, Type: {selected_model_type}"
            )

            # Hide speaker dropdown for VoiceDesign and Base models
            if selected_model_type == "VoiceDesign":
                speaker_info_text = "Voice Design model - describe your desired voice in the Instruction field"
                print(
                    f"[load_model_handler] VoiceDesign mode - hiding speaker dropdown"
                )
                return (
                    status,
                    gr.update(choices=lang_choices, value="English", interactive=True),
                    gr.update(visible=False),
                    speaker_info_text,
                    gr.update(interactive=True),
                    "",
                    None,
                    gr.update(visible=False),  # ref_audio_input
                    gr.update(visible=False),  # ref_text_input
                )
            elif selected_model_type == "Base":
                speaker_info_text = (
                    "Voice Clone model - upload reference audio to clone voice"
                )
                print(
                    f"[load_model_handler] Base mode - hiding speaker dropdown, showing voice clone UI"
                )
                return (
                    status,
                    gr.update(choices=lang_choices, value="English", interactive=True),
                    gr.update(visible=False),
                    speaker_info_text,
                    gr.update(interactive=True),
                    "",
                    None,
                    gr.update(visible=True),  # ref_audio_input
                    gr.update(visible=True),  # ref_text_input
                )
            else:
                print(
                    f"[load_model_handler] CustomVoice mode - showing speaker dropdown"
                )
                return (
                    status,
                    gr.update(choices=lang_choices, value="English", interactive=True),
                    gr.update(
                        choices=speaker_choices,
                        value="Aiden",
                        interactive=True,
                        visible=True,
                    ),
                    speaker_descriptions.get("aiden", ""),
                    gr.update(interactive=True),
                    "",
                    None,
                    gr.update(visible=False),  # ref_audio_input
                    gr.update(visible=False),  # ref_text_input
                )
        else:
            return (
                status,
                gr.update(choices=[], value=None, interactive=False),
                gr.update(choices=[], value=None, interactive=False),
                "",
                gr.update(interactive=False),
                "",
                None,
                gr.update(visible=False),  # ref_audio_input
                gr.update(visible=False),  # ref_text_input
            )

    with gr.Blocks(
        title="Qwen3-TTS",
    ) as demo:
        gr.Markdown("# 🎙️ Qwen3-TTS")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Model Setup")

                if not AVAILABLE_MODELS:
                    gr.Markdown(
                        "❌ **No models found**. Please check MODELS_BASE_DIR environment variable."
                    )
                    model_dropdown = gr.Dropdown(
                        label="Model",
                        choices=[],
                        value=None,
                        interactive=False,
                    )
                else:
                    model_dropdown = gr.Dropdown(
                        label="Model",
                        choices=list(AVAILABLE_MODELS.keys()),
                        value=list(AVAILABLE_MODELS.keys())[0],
                        info="Select TTS model.",
                    )

                available_devices = get_available_devices()
                device_dropdown = gr.Dropdown(
                    label="Device",
                    choices=list(available_devices.keys()),
                    value=list(available_devices.keys())[0],
                    info="Select compute device (CPU or GPU).",
                )

                load_model_btn = gr.Button(
                    "📥 Load Model", variant="primary", size="lg"
                )

                model_status = gr.Textbox(
                    value="⚠️ Model not loaded",
                    interactive=False,
                    show_label=False,
                )

                gr.Markdown("### 🎤 Voice Settings")

                language_dropdown = gr.Dropdown(
                    label="Language",
                    choices=[],
                    value=None,
                    interactive=False,
                    info="Select language or use Auto for automatic detection.",
                    allow_custom_value=True,
                )

                speaker_dropdown = gr.Dropdown(
                    label="Speaker",
                    choices=[],
                    value=None,
                    interactive=False,
                    info="Select voice profile (not used for VoiceDesign).",
                    allow_custom_value=True,
                    visible=True,
                )

                speaker_info = gr.Textbox(
                    value="",
                    interactive=False,
                    show_label=False,
                )

                instruct_input = gr.Textbox(
                    label="Instruction (Optional)",
                    value="Speak in an incredulous tone with panic.",
                    placeholder="e.g., 用特别愤怒的语气说 / Speak in an angry tone",
                    lines=2,
                    info="Optional instruction to control tone, emotion, or speaking style.",
                )

                ref_audio_input = gr.Audio(
                    label="Reference Audio (Voice Clone)",
                    type="filepath",
                    visible=False,
                )

                ref_text_input = gr.Textbox(
                    label="Reference Text (Voice Clone)",
                    value="",
                    placeholder="Enter the transcript of the reference audio...",
                    lines=2,
                    visible=False,
                )

                gr.Markdown("""
                **Supported Models:**
                - Qwen3-TTS-12Hz-1.7B-CustomVoice
                - Qwen3-TTS-12Hz-0.6B-CustomVoice
                - Qwen3-TTS-12Hz-1.7B-VoiceDesign
                - Qwen3-TTS-12Hz-0.6B-VoiceDesign
                - Qwen3-TTS-12Hz-1.7B-Base
                - Qwen3-TTS-12Hz-0.6B-Base

                **Languages:** 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
                """)

            with gr.Column(scale=1):
                gr.Markdown("### 🔊 Output")

                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="numpy",
                )

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

                gr.Markdown("### 📝 Input")

                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    value="It's in the top drawer... wait, it's empty? No way, that's impossible!",
                    placeholder="Enter text here...",
                    lines=4,
                    info="Enter the text you want to convert to speech.",
                )

                generate_btn = gr.Button(
                    "🎵 Generate Speech",
                    variant="primary",
                    size="lg",
                    interactive=False,
                )

                voice_design_tips = gr.Markdown(
                    """
                    ### 🎨 Voice Design Instructions

                    For **VoiceDesign** models, describe voice characteristics in detail:

                    **What to describe:**
                    - **Gender**: Male, female, child, etc.
                    - **Age range**: Teen, young adult, middle-aged, etc.
                    - **Voice qualities**: Pitch (high/low), timbre, breathiness, resonance
                    - **Speaking style**: Energetic, calm, nervous, confident, etc.
                    - **Emotional tone**: Happy, sad, angry, excited, etc.
                    - **Regional accent**: American, British, etc.

                    **Example instructions:**
                    - Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous
                    - Warm, gentle middle-aged female voice with a slightly breathy quality and soft consonants
                    - Energetic young male voice with bright timbre, slightly fast pace, and upbeat intonation
                    - Young female voice with high pitch and obvious fluctuations, creating a clingy, artificial and deliberately cute effect
                    - Mature and steady middle-aged male voice, moderate speed, calm tone, with a slight magnetism
                    - Vigorous young female voice, fast speed, bright and infectious
                    """,
                    visible=is_voice_design_visible,
                )

                custom_voice_tips = gr.Markdown(
                    """
                    ### 🎤 Custom Voice Instructions

                    For **CustomVoice** models, use the instruction field to modify the selected speaker's voice:

                    **Available Speakers:**
                    - **Vivian**: Bright, slightly edgy young female voice. (Native: Chinese)
                    - **Serena**: Warm, gentle young female voice. (Native: Chinese)
                    - **Uncle_Fu**: Seasoned male voice with a low, mellow timbre. (Native: Chinese)
                    - **Dylan**: Youthful Beijing male voice with a clear, natural timbre. (Native: Chinese, Beijing Dialect)
                    - **Eric**: Lively Chengdu male voice with a slightly husky brightness. (Native: Chinese, Sichuan Dialect)
                    - **Ryan**: Dynamic male voice with strong rhythmic drive. (Native: English)
                    - **Aiden**: Sunny American male voice with a clear midrange. (Native: English)
                    - **Ono_Anna**: Playful Japanese female voice with a light, nimble timbre. (Native: Japanese)
                    - **Sohee**: Warm Korean female voice with rich emotion. (Native: Korean)

                    **Instruction Examples:**
                    - Speak in an angry tone
                    - Speak in a cute, childish tone
                    - Very happy
                    - Speak in an incredulous tone with panic
                    - Speak in a bright, cheerful tone
                    - Speak with sadness in your voice
                    - Speak in an excited, energetic tone
                    - Speak in an extremely angry tone
                    - Speak in a sad tone
                    """,
                    visible=is_custom_voice_visible,
                )

                voice_clone_tips = gr.Markdown(
                    """
                    ### 🔊 Voice Clone Instructions

                    For **Base** models, clone voices from reference audio clips:

                    **Requirements:**
                    - **Reference Audio**: Upload a 3+ second audio file of the voice you want to clone
                    - **Reference Text**: Provide the exact transcript of the reference audio

                    **Best Practices:**
                    - Use clear, high-quality reference audio (3-10 seconds works best)
                    - Provide accurate transcription of the reference audio
                    - The reference audio should represent the voice characteristics you want to clone
                    - Avoid noisy or distorted audio for best results

                    **Supported Formats:**
                    - Audio files: WAV, MP3, FLAC, etc.
                    - The model will extract speaker embedding and style from the reference
                    """,
                    visible=is_voice_clone_visible,
                )

        # Connect load model button
        load_model_btn.click(
            fn=load_model_handler,
            inputs=[model_dropdown, device_dropdown],
            outputs=[
                model_status,
                language_dropdown,
                speaker_dropdown,
                speaker_info,
                generate_btn,
                status_output,
                audio_output,
                ref_audio_input,
                ref_text_input,
            ],
        )

        # Connect model dropdown to show/hide tips
        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[
                voice_design_tips,
                custom_voice_tips,
                voice_clone_tips,
                ref_audio_input,
                ref_text_input,
                generate_btn,
            ],
            show_progress=False,
        ).then(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[
                voice_design_tips,
                custom_voice_tips,
                voice_clone_tips,
                ref_audio_input,
                ref_text_input,
                generate_btn,
            ],
            show_progress=False,
        )

        # Connect speaker dropdown to info textbox
        speaker_dropdown.change(
            fn=update_speaker_info,
            inputs=[speaker_dropdown],
            outputs=[speaker_info],
            show_progress=False,
        )

        # Connect generate button
        generate_btn.click(
            fn=generate_speech,
            inputs=[
                text_input,
                language_dropdown,
                speaker_dropdown,
                instruct_input,
                ref_audio_input,
                ref_text_input,
            ],
            outputs=[audio_output, status_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
