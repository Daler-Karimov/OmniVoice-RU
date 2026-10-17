# OmniVoice-RU

[Русский](#русский) | [English](#english)

---

## Русский

Веб-интерфейс (WebUI) для многоязычной модели синтеза и клонирования речи **OmniVoice (k2-fsa)** с поддержкой квантизированных моделей **GGUF (BF16, Q8_0, Q4_K_M)** и оригинального **PyTorch FP16**.

### Возможности

- **Выбор модели на лету:**
  - **GGUF Q8_0** (рекомендуемый 8-битный квант, ~620 МБ, высокое качество, инференс ~0.4 сек на NVIDIA RTX).
  - **GGUF Q4_K_M** (ультра-легкий 4-битный квант, ~384 МБ, минимальное потребление VRAM).
  - **GGUF BF16** (16-битный bfloat16, ~1.17 ГБ, исходная точность без квантования).
  - **PyTorch FP16** (официальная модель с Hugging Face, стандартный рантайм Transformers).
- **Дизайн голоса (Voice Design):**
  - Управление полом (мужской, женский).
  - Управление возрастом (ребенок, подросток, молодой, средний, пожилой).
  - Высота тона (очень низкий, низкий, средний, высокий, очень высокий).
  - Стили (обычный, шепот).
  - Акценты (русский, американский, британский, австралийский и др.).
- **Клонирование голоса (Voice Cloning):**
  - Копирование тембра по короткому аудио-образцу (3-10 секунд речи) и тексту транскрипта.
- **Охват языков:**
  - Поддержка более 646 языков и диалектов.
- **Публичный доступ:**
  - Автоматическое создание публичной ссылки Gradio (share link) для удаленного доступа.
- **Быстрый запуск:**
  - Скрипт `start_webui.bat` для запуска в один клик под Windows с автоматическим созданием `venv` и установкой зависимостей.

### Быстрый старт (Windows)

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Daler-Karimov/OmniVoice-RU.git
cd OmniVoice-RU
```

2. Запустите батник:
```cmd
start_webui.bat
```
При первом запуске скрипт автоматически создаст виртуальное окружение `venv`, установит PyTorch с поддержкой CUDA и все необходимые библиотеки, после чего откроет веб-интерфейс в браузере.

3. Веб-интерфейс будет доступен по локальному адресу и по публичной ссылке:
```
Локально: http://127.0.0.1:7860
Публично: https://xxxx.gradio.live (ссылка выводится в окне консоли)
```
Для запуска только локально (без публичной ссылки) используйте:
```cmd
start_webui.bat --no-share
```

### Сборка C++ движка omnivoice.cpp (для GGUF моделей)

В репозиторий включен исходный код легковесного C++17 рантайма на базе GGML (`omnivoice_cpp`).

Для пересборки под вашу версию CUDA:
```cmd
cd omnivoice_cpp
build_win.bat
```
Скрипт скомпилирует бинарники `omnivoice-tts.exe` и `ggml-cuda.dll` с использованием Visual Studio и Ninja.

### Структура проекта

```
OmniVoice-RU/
|-- webui.py            # Главный Gradio веб-интерфейс с выбором моделей
|-- start_webui.bat     # Скрипт запуска в один клик под Windows
|-- run.bat             # Дополнительный ярлык запуска
|-- pyproject.toml      # Конфигурация зависимостей Python
|-- models/             # Папка для GGUF моделей (автоматически скачиваются при выборе)
|-- omnivoice/          # Исходный код Python-пакета OmniVoice
`-- omnivoice_cpp/      # C++ рантайм на базе GGML (omnivoice.cpp)
```

### Лицензия и первоисточники

- Оригинальная модель и код: [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (Apache 2.0 / CC-BY-NC 4.0).
- Движок GGML и квантизация: [ServeurpersoCom/omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp) и [Serveurperso/OmniVoice-GGUF](https://huggingface.co/Serveurperso/OmniVoice-GGUF).

---

## English

WebUI for the multilingual text-to-speech and voice cloning model **OmniVoice (k2-fsa)**, featuring support for **GGUF quantized models (BF16, Q8_0, Q4_K_M)** and the original **PyTorch FP16** checkpoint.

### Key Features

- **On-the-fly Model Switching:**
  - **GGUF Q8_0** (recommended 8-bit quantization, ~620 MB, high quality, ~0.4s inference on NVIDIA RTX).
  - **GGUF Q4_K_M** (ultra-lightweight 4-bit quantization, ~384 MB, lowest VRAM consumption).
  - **GGUF BF16** (16-bit bfloat16, ~1.17 GB, original precision without quantization).
  - **PyTorch FP16** (official Hugging Face model, standard Transformers runtime).
- **Voice Design:**
  - Gender control (male, female).
  - Age control (child, teenager, young adult, middle-aged, elderly).
  - Pitch adjustment (very low pitch, low pitch, moderate pitch, high pitch, very high pitch).
  - Styles (normal, whisper).
  - Accents (Russian, American, British, Australian, and more).
- **Voice Cloning:**
  - Accurate timbre cloning using a short reference audio clip (3-10 seconds) and its transcript text.
- **Language Coverage:**
  - Over 646 languages and dialects supported.
- **Public Sharing:**
  - Automatic Gradio public share link generation for remote access.
- **One-Click Launch:**
  - `start_webui.bat` script for Windows with automatic virtual environment (`venv`) setup and dependency installation.

### Quick Start (Windows)

1. Clone the repository:
```bash
git clone https://github.com/Daler-Karimov/OmniVoice-RU.git
cd OmniVoice-RU
```

2. Run the batch script:
```cmd
start_webui.bat
```
On the first run, the script automatically creates a `venv` virtual environment, installs PyTorch with CUDA support, installs all dependencies, and opens the WebUI in your default browser.

3. The WebUI is accessible locally and publicly:
```
Local URL:  http://127.0.0.1:7860
Public URL: https://xxxx.gradio.live (displayed in the console window)
```
To run locally only (without creating a public link):
```cmd
start_webui.bat --no-share
```

### Building the C++ Engine omnivoice.cpp (for GGUF models)

This repository includes the source code for the lightweight GGML-based C++17 runtime (`omnivoice_cpp`).

To build with CUDA support on Windows:
```cmd
cd omnivoice_cpp
build_win.bat
```
This compiles `omnivoice-tts.exe` and `ggml-cuda.dll` using Visual Studio and Ninja.

### Project Structure

```
OmniVoice-RU/
|-- webui.py            # Main Gradio WebUI with model selection
|-- start_webui.bat     # One-click Windows launch script
|-- run.bat             # Alternative launcher shortcut
|-- pyproject.toml      # Python dependency configuration
|-- models/             # Directory for GGUF models (auto-downloaded on demand)
|-- omnivoice/          # Python OmniVoice source package
`-- omnivoice_cpp/      # C++ GGML-based runtime (omnivoice.cpp)
```

### License and Credits

- Original model and code: [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (Apache 2.0 / CC-BY-NC 4.0).
- GGML engine and quantization: [ServeurpersoCom/omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp) and [Serveurperso/OmniVoice-GGUF](https://huggingface.co/Serveurperso/OmniVoice-GGUF).
