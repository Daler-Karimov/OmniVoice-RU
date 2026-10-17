# -*- coding: utf-8 -*-
"""
OmniVoice WebUI with Custom Model & GGUF Quantization Selection (BF16, Q8_0, Q4_K_M, PyTorch FP16).
CUDA-accelerated inference.
"""

import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Optional

import gradio as gr
import numpy as np
import soundfile as sf
import torch

# Base paths
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
CPP_BUILD_DIR = ROOT_DIR / "omnivoice_cpp" / "build"
CPP_EXE = CPP_BUILD_DIR / "omnivoice-tts.exe"

# Model definitions
MODEL_VARIANTS = {
    "GGUF Q8_0 (Рекомендуемый, ~620 МБ, высокая скорость)": {
        "type": "gguf",
        "model": MODELS_DIR / "omnivoice-base-Q8_0.gguf",
        "codec": MODELS_DIR / "omnivoice-tokenizer-Q8_0.gguf",
        "desc": "8-битное квантование GGUF через omnivoice.cpp (CUDA). Оптимальный баланс качества и скорости.",
    },
    "GGUF Q4_K_M (Ультра-легкий, ~384 МБ, минимальный VRAM)": {
        "type": "gguf",
        "model": MODELS_DIR / "omnivoice-base-Q4_K_M.gguf",
        "codec": MODELS_DIR / "omnivoice-tokenizer-Q4_K_M.gguf",
        "desc": "4-битное квантование GGUF. Минимальное потребление памяти (~380 МБ) и максимальная скорость.",
    },
    "GGUF BF16 (Максимальная точность, ~1.17 ГБ)": {
        "type": "gguf",
        "model": MODELS_DIR / "omnivoice-base-BF16.gguf",
        "codec": MODELS_DIR / "omnivoice-tokenizer-BF16.gguf",
        "desc": "16-битный bfloat16 без квантования через C++ движок GGML.",
    },
    "PyTorch FP16 (Оригинал k2-fsa/OmniVoice, ~3.03 ГБ)": {
        "type": "pytorch",
        "repo_id": "k2-fsa/OmniVoice",
        "desc": "Официальная модель Hugging Face, стандартный рантайм PyTorch.",
    },
}

DEFAULT_MODEL = "GGUF Q8_0 (Рекомендуемый, ~620 МБ, высокая скорость)"

# Lazy loaded PyTorch model
_pytorch_model = None

def get_pytorch_model():
    global _pytorch_model
    if _pytorch_model is None:
        print("[+] Загрузка PyTorch модели k2-fsa/OmniVoice на CUDA...")
        from omnivoice import OmniVoice
        _pytorch_model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
            load_asr=True
        )
        print("[+] PyTorch модель успешно загружена!")
    return _pytorch_model

# Supported language list
try:
    from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
    ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
except Exception:
    ALL_LANGUAGES = ["Auto", "Russian", "English", "Chinese", "Spanish", "French", "German", "Japanese", "Korean"]

# Mappings for Voice Design (only strictly supported keywords)
GENDER_MAP = {
    "Не указано": "",
    "Мужской (Male)": "male",
    "Женский (Female)": "female",
}

AGE_MAP = {
    "Не указано": "",
    "Ребенок (Child)": "child",
    "Подросток (Teenager)": "teenager",
    "Молодой (Young Adult)": "young adult",
    "Средний возраст (Middle-aged)": "middle-aged",
    "Пожилой (Elderly)": "elderly",
}

PITCH_MAP = {
    "Не указано": "",
    "Очень низкий (Very Low Pitch)": "very low pitch",
    "Низкий (Low Pitch)": "low pitch",
    "Средний (Moderate Pitch)": "moderate pitch",
    "Высокий (High Pitch)": "high pitch",
    "Очень высокий (Very High Pitch)": "very high pitch",
}

STYLE_MAP = {
    "Обычный (Normal)": "",
    "Шепот (Whisper)": "whisper",
}

ACCENT_MAP = {
    "Не указано": "",
    "Русский акцент (Russian Accent)": "russian accent",
    "Американский акцент (American Accent)": "american accent",
    "Британский акцент (British Accent)": "british accent",
    "Австралийский акцент (Australian Accent)": "australian accent",
    "Канадский акцент (Canadian Accent)": "canadian accent",
    "Китайский акцент (Chinese Accent)": "chinese accent",
    "Индийский акцент (Indian Accent)": "indian accent",
    "Японский акцент (Japanese Accent)": "japanese accent",
    "Корейский акцент (Korean Accent)": "korean accent",
    "Португальский акцент (Portuguese Accent)": "portuguese accent",
}


def build_instruction(gender_key, age_key, pitch_key, style_key, accent_key, custom_instruct=""):
    parts = []
    g = GENDER_MAP.get(gender_key, "")
    if g:
        parts.append(g)
    a = AGE_MAP.get(age_key, "")
    if a:
        parts.append(a)
    p = PITCH_MAP.get(pitch_key, "")
    if p:
        parts.append(p)
    s = STYLE_MAP.get(style_key, "")
    if s:
        parts.append(s)
    ac = ACCENT_MAP.get(accent_key, "")
    if ac:
        parts.append(ac)
    if custom_instruct and custom_instruct.strip():
        parts.append(custom_instruct.strip().lower())
    return ", ".join(parts)


def generate_speech(
    model_choice: str,
    mode: str,
    text: str,
    language: str,
    gender: str,
    age: str,
    pitch: str,
    style: str,
    accent: str,
    custom_instruct: str,
    ref_audio: Optional[str],
    ref_text: Optional[str],
    steps: int,
    seed: int,
):
    if not text or not text.strip():
        return None, "Пожалуйста, введите текст для синтеза речи."

    text = text.strip()
    config = MODEL_VARIANTS.get(model_choice, MODEL_VARIANTS[DEFAULT_MODEL])
    instruction = build_instruction(gender, age, pitch, style, accent, custom_instruct)
    lang_arg = language if (language and language != "Auto") else "None"

    t0 = time.time()

    # 1. GGUF Engine via omnivoice-tts.exe
    if config["type"] == "gguf":
        model_path = config["model"]
        codec_path = config["codec"]

        if not model_path.exists() or not codec_path.exists():
            print(f"[+] Скачивание {model_path.name} с Hugging Face...")
            try:
                from huggingface_hub import hf_hub_download
                os.makedirs(MODELS_DIR, exist_ok=True)
                if not model_path.exists():
                    hf_hub_download(repo_id="Serveurperso/OmniVoice-GGUF", filename=model_path.name, local_dir=str(MODELS_DIR))
                if not codec_path.exists():
                    hf_hub_download(repo_id="Serveurperso/OmniVoice-GGUF", filename=codec_path.name, local_dir=str(MODELS_DIR))
            except Exception as e:
                return None, f"Файлы модели {model_path.name} не найдены и не удалось их скачать: {e}"

        if not CPP_EXE.exists():
            return None, f"Исполняемый файл {CPP_EXE} не найден."

        out_wav = Path(tempfile.gettempdir()) / f"omnivoice_{int(time.time()*1000)}.wav"

        cmd = [
            str(CPP_EXE),
            "--model", str(model_path),
            "--codec", str(codec_path),
            "--device", "CUDA0" if torch.cuda.is_available() else "CPU",
            "--steps", str(int(steps)),
            "--seed", str(int(seed) if seed is not None and seed >= 0 else -1),
            "-o", str(out_wav),
        ]

        if lang_arg != "None":
            cmd.extend(["--lang", lang_arg])

        if instruction:
            cmd.extend(["--instruct", instruction])

        ref_text_tmp = None
        if mode == "clone":
            if not ref_audio:
                return None, "В режиме клонирования необходимо загрузить референсное аудио."
            cmd.extend(["--ref-wav", str(ref_audio)])
            if ref_text and ref_text.strip():
                ref_text_tmp = Path(tempfile.gettempdir()) / f"ref_text_{int(time.time()*1000)}.txt"
                ref_text_tmp.write_text(ref_text.strip(), encoding="utf-8")
                cmd.extend(["--ref-text", str(ref_text_tmp)])
            else:
                return None, "Для GGUF движка требуется указать текст референсного аудио (транскрипт)."

        try:
            res = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8", errors="ignore")
            return None, f"Ошибка omnivoice.cpp:\n{err_msg}"
        finally:
            if ref_text_tmp and ref_text_tmp.exists():
                try:
                    ref_text_tmp.unlink()
                except Exception:
                    pass

        if not out_wav.exists():
            return None, "Выходной аудиофайл не был создан."

        data, sr = sf.read(str(out_wav), dtype="float32")
        try:
            out_wav.unlink()
        except Exception:
            pass

        dur = len(data) / sr
        elapsed = time.time() - t0
        rtf = elapsed / dur if dur > 0 else 0
        info = f"Синтезировано за {elapsed:.2f} сек. Длина аудио: {dur:.2f} сек (RTF: {rtf:.3f}) на движке {model_choice}."
        return (sr, data), info

    # 2. PyTorch Engine
    else:
        pt_model = get_pytorch_model()
        from omnivoice import OmniVoiceGenerationConfig
        gen_config = OmniVoiceGenerationConfig(num_step=int(steps))

        kw = {
            "text": text,
            "language": language if language != "Auto" else None,
            "generation_config": gen_config,
        }
        if instruction:
            kw["instruct"] = instruction

        if mode == "clone":
            if not ref_audio:
                return None, "В режиме клонирования необходимо загрузить референсное аудио."
            kw["ref_audio"] = ref_audio
            if ref_text and ref_text.strip():
                kw["ref_text"] = ref_text.strip()

        try:
            audio_list = pt_model.generate(**kw)
            waveform = audio_list[0]
            sr = pt_model.sampling_rate
            dur = len(waveform) / sr
            elapsed = time.time() - t0
            rtf = elapsed / dur if dur > 0 else 0
            info = f"Синтезировано за {elapsed:.2f} сек. Длина: {dur:.2f} сек (RTF: {rtf:.3f}) через PyTorch FP16."
            return (sr, waveform), info
        except Exception as e:
            return None, f"Ошибка генерации PyTorch: {e}"


# --- GRADIO ИНТЕРФЕЙС ---
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

with gr.Blocks(title="OmniVoice WebUI") as demo:
    gr.Markdown(
        """
        # OmniVoice WebUI (k2-fsa)
        ### Модель синтеза и клонирования речи (600+ языков)
        **Поддержка квантизированных GGUF (BF16, Q8_0, Q4_K_M) с ускорением NVIDIA CUDA и стандартного PyTorch.**
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            model_dropdown = gr.Dropdown(
                label="Выбор модели (GGUF квантование или PyTorch)",
                choices=list(MODEL_VARIANTS.keys()),
                value=DEFAULT_MODEL,
                interactive=True,
            )
            model_info_text = gr.Markdown(value=MODEL_VARIANTS[DEFAULT_MODEL]["desc"])

    def on_model_change(choice):
        return MODEL_VARIANTS.get(choice, {}).get("desc", "")

    model_dropdown.change(on_model_change, inputs=[model_dropdown], outputs=[model_info_text])

    with gr.Tabs():
        # --- ВКЛАДКА 1: ДИЗАЙН ГОЛОСА ---
        with gr.TabItem("Дизайн голоса (Text-to-Speech)"):
            with gr.Row():
                with gr.Column(scale=5):
                    vd_text = gr.Textbox(
                        label="Текст для синтеза речи",
                        placeholder="Введите текст для синтеза речи...",
                        lines=4,
                        value="Привет! Это тест модели OmniVoice с возможностью выбора квантования GGUF.",
                    )
                    vd_lang = gr.Dropdown(
                        label="Язык (Language)",
                        choices=ALL_LANGUAGES,
                        value="Russian",
                        interactive=True,
                    )

                    with gr.Accordion("Параметры голоса (Voice Design)", open=True):
                        with gr.Row():
                            vd_gender = gr.Dropdown(label="Пол", choices=list(GENDER_MAP.keys()), value="Женский (Female)")
                            vd_age = gr.Dropdown(label="Возраст", choices=list(AGE_MAP.keys()), value="Молодой (Young Adult)")
                        with gr.Row():
                            vd_pitch = gr.Dropdown(label="Высота тона", choices=list(PITCH_MAP.keys()), value="Не указано")
                            vd_style = gr.Dropdown(label="Стиль", choices=list(STYLE_MAP.keys()), value="Обычный (Normal)")
                        with gr.Row():
                            vd_accent = gr.Dropdown(label="Акцент", choices=list(ACCENT_MAP.keys()), value="Не указано")
                        vd_custom = gr.Textbox(
                            label="Дополнительная инструкция на английском (необязательно)",
                            placeholder="например: female, young adult, whisper",
                            lines=1,
                        )

                    with gr.Accordion("Настройки генерации", open=False):
                        with gr.Row():
                            vd_steps = gr.Slider(label="Шаги диффузии (Steps)", minimum=16, maximum=64, value=32, step=4)
                            vd_seed = gr.Number(label="Сид (Seed, -1 для случайного)", value=-1, precision=0)

                    vd_btn = gr.Button("Синтезировать речь", variant="primary", size="lg")

                with gr.Column(scale=4):
                    vd_audio = gr.Audio(label="Результат синтеза", type="numpy", interactive=False)
                    vd_status = gr.Textbox(label="Статус генерации", interactive=False)

            vd_btn.click(
                fn=lambda m, t, l, g, a, p, s, ac, c, st, se: generate_speech(
                    m, "design", t, l, g, a, p, s, ac, c, None, None, st, se
                ),
                inputs=[
                    model_dropdown, vd_text, vd_lang, vd_gender, vd_age, vd_pitch, vd_style, vd_accent, vd_custom, vd_steps, vd_seed
                ],
                outputs=[vd_audio, vd_status],
            )

        # --- ВКЛАДКА 2: КЛОНИРОВАНИЕ ГОЛОСА ---
        with gr.TabItem("Клонирование голоса (Voice Cloning)"):
            with gr.Row():
                with gr.Column(scale=5):
                    vc_text = gr.Textbox(
                        label="Текст для произнесения клонированным голосом",
                        placeholder="Введите текст...",
                        lines=3,
                        value="Здравствуйте! Мой голос был успешно скопирован с аудиозаписи.",
                    )
                    vc_lang = gr.Dropdown(
                        label="Язык (Language)",
                        choices=ALL_LANGUAGES,
                        value="Russian",
                        interactive=True,
                    )
                    vc_ref_audio = gr.Audio(
                        label="Референсная аудиозапись (3-10 секунд чистой речи)",
                        type="filepath",
                    )
                    vc_ref_text = gr.Textbox(
                        label="Транскрипт референса (точный текст из аудио)",
                        placeholder="Введите, что именно говорится в референсном аудио...",
                        lines=2,
                    )
                    with gr.Accordion("Настройки генерации", open=False):
                        with gr.Row():
                            vc_steps = gr.Slider(label="Шаги диффузии (Steps)", minimum=16, maximum=64, value=32, step=4)
                            vc_seed = gr.Number(label="Сид (Seed, -1 для случайного)", value=-1, precision=0)

                    vc_btn = gr.Button("Клонировать голос и озвучить", variant="primary", size="lg")

                with gr.Column(scale=4):
                    vc_audio = gr.Audio(label="Результат клонирования", type="numpy", interactive=False)
                    vc_status = gr.Textbox(label="Статус генерации", interactive=False)

            vc_btn.click(
                fn=lambda m, t, l, ra, rt, st, se: generate_speech(
                    m, "clone", t, l, "Не указано", "Не указано", "Не указано", "Обычный (Normal)", "Не указано", "", ra, rt, st, se
                ),
                inputs=[
                    model_dropdown, vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_steps, vc_seed
                ],
                outputs=[vc_audio, vc_status],
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OmniVoice WebUI")
    parser.add_argument("--share", action="store_true", default=True, help="Создавать публичную ссылку Gradio")
    parser.add_argument("--no-share", action="store_false", dest="share", help="Отключить публичную ссылку Gradio")
    parser.add_argument("--ip", default="0.0.0.0", help="IP адрес сервера (по умолчанию 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Порт сервера (по умолчанию 7860)")
    args, _ = parser.parse_known_args()

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=theme,
    )
