# Native Python bindings for kitten-tts (`kitten_tts_cpp`)

This builds the `kitten_tts_cpp` Python package from this repo, compiled for the CPU it is built on (`-march=native`) and linked against CPU-only LibTorch. It is a drop-in replacement for the PyPI wheel `kitten-tts-cpp==0.2.0`: same class, same signatures, same Python code. The only difference is the compiled part.

Result on this machine: 45.8 generated tokens/s through Python, the same as the native CLI (45.5), and the same tokens bit-for-bit for a fixed seed.

## 1. What "bindings" are, and how this module plugs in

Python cannot call C++ functions directly. A *binding module* is a shared library (`.so`) that Python can `import`. It holds the C++ code plus glue that converts Python values (dict, str, numpy array) to C++ values and back. Here the glue is written with [pybind11](https://pybind11.readthedocs.io), the same tool the PyPI wheel used (its `.so` contains `pybind11::` symbols).

The PyPI package has three parts:

```
kitten_tts_cpp/__init__.py   pure Python, re-exports KittenTTS
kitten_tts_cpp/model.py      pure Python: argument checks, threading, streaming queue
kitten_tts_cpp/_native.so    compiled: the actual engine
```

`model.py` uses only this small `_native` interface (found by reading it):

| Python call in `model.py` | What it must do | C++ side |
|---|---|---|
| `_native.Engine(args)` | `args` is a `dict[str, str]` of CLI flags (`--threads`, `--decoder`, `--data`, ...). Load everything once. | `kitten::engine::engine` |
| `engine.metadata()` | JSON string `{"config": ..., "voices": [...]}` | `engine::metadata` |
| `engine.normalize(text)` | normalized text | `engine::normalize` |
| `engine.generate(args, callback)` | returns `(float32 numpy array, report JSON string)`. `callback` is `None` or a function called with each chunk's audio; it returns `False` to stop. | `engine::generate` |

So I kept `model.py` and `__init__.py` **byte-for-byte unchanged** (sha256 matches the wheel's RECORD) and wrote a new `_native` with that interface. Because the Python layer is the original, the public API matches automatically.

```
your code / server.py
        |
        v
kitten_tts_cpp.KittenTTS        (model.py, original, unchanged)
        |  dict of "--flag": "value"
        v
_native.Engine                   (python/native.cpp, pybind11 glue)
        |
        v
kitten::engine                   (engine.cpp: load once, generate many)
   |            |             |
   v            v             v
llama/ggml   kitten-text   kitten-decoder (LibTorch TorchScript)
(LM, -march=native)  (normalize, split, join)
```

## 2. Files added or changed

All in this repo unless noted.

| File | Status | Why |
|---|---|---|
| `tools/kitten-tts/engine.h`, `engine.cpp` | new | The CLI did everything inside `main()`. This class does the same steps, but loads the model, decoder, normalizer and voices **once** in the constructor, and `generate()` returns audio instead of writing a WAV. The per-chunk loop (prompt build, token budget, prefill, `kitten::process` + `kitten::sample`, decoder, `kitten::join`) is copied from `main.cpp` line for line. No sampling, quantization or kernel code is changed. CLI-only features (`--logits`, `--force-tokens`, `--greedy`, `--tokens-only`, `--repeat`) are left out. It also takes the two flags `model.py` sends that the CLI does not have: `--use-reference 0/1` and `--use-emotion 0/1` (unset = the CLI's defaults). |
| `tools/kitten-tts/assets.cpp` | 1 line changed | Accepts `--hf-token` (from `KittenTTS(hf_token=...)`), with the `HF_TOKEN` env var as fallback. The CLI never passes this key, so the CLI's behavior is the same. |
| `tools/kitten-tts/python/native.cpp` | new | The pybind11 module `_native`. It releases the Python GIL during load and generation, so other Python threads (the server's event loop) keep running, and takes the GIL back only to call the stream callback. Audio is handed to numpy without a copy. |
| `tools/kitten-tts/python/kitten_tts_cpp/__init__.py`, `model.py` | new (vendored) | Copied unchanged from the MIT-licensed `kitten_tts_cpp-0.2.0` wheel (its LICENSE is this repo's ggml MIT license). |
| `tools/kitten-tts/python/pyproject.toml` | new | Minimal setuptools metadata so `pip install` can install the staged package. **torch is deliberately not a dependency**: pip would pull the CUDA wheel from PyPI. |
| `tools/kitten-tts/CMakeLists.txt` | changed (additions only) | New option `KITTEN_TTS_BUILD_PYTHON` (default OFF). When ON it builds `_native` and stages a complete package in `<build>/tools/kitten-tts/python/`: the `.so`, the Python files, `_build_info.json` and the text-normalizer `data/` directory. |
| `~/patchedkitten/KittenTTS/kittenml/kittentts2/docker/server.py` | changed (other repo, not committed) | `from kittenml import KittenTTS` moved into the non-`cpp` branch, so the cpp backend does not import the PyTorch model code. |
| `~/patchedkitten/KittenTTS/kittenml/kittentts2/__init__.py` | changed (other repo, not committed) | Made lazy (a module `__getattr__`). `python -m kittenml.kittentts2.docker.server` always runs this file first, and it imported `.model`, which pulls in transformers, librosa, safetensors, and more. Now names import on first use. With the cpp backend only fastapi, uvicorn, pydantic, numpy and soundfile are needed. |

The CLI (`main.cpp`) is untouched. A later cleanup could make `main()` call `kitten::engine` so the loop exists only once. I did not do that here, to keep the CLI's behavior untouched.

### Why a separate static build directory (`build-py/`)

Your `build/` has `BUILD_SHARED_LIBS=ON`, so the CLI needs `libllama.so`, `libggml*.so` and so on next to it. A Python module built that way would need those files copied and found at runtime too. With `-DBUILD_SHARED_LIBS=OFF` all of llama/ggml/text-processing is linked *into* `_native.so`, the same as the PyPI wheel. Static code that goes into a shared library must be compiled as position-independent code, hence `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`. CMake prints a warning if you turn on the Python option without these two.

### Why "build with CMake, then pip install the staged folder" (and not a CMake-driven pip build)

A `pip install .` that runs CMake itself (scikit-build) rebuilds all of llama + ggml in a temporary directory every time and hides compiler errors in pip's log. Two visible steps are easier to understand and debug: CMake builds and stages the package, then pip only copies files.

### The torch pin

`model.py` reads `_build_info.json` and refuses to import if the installed torch version differs (C++ code built against one LibTorch version is not guaranteed to work with another). CMake writes this file from the torch it actually found:

```json
{
  "torch_version": "2.14.1",
  "portable": false,
  "ggml_native": "ON",
  "cxx_flags": ""
}
```

**This build needs `torch==2.14.1+cpu`** (from `https://download.pytorch.org/whl/cpu`). If you upgrade torch, rebuild `_native`.

## 3. Build and install (exact commands)

Inside the `kitten` toolbox.

One-time setup:

```sh
sudo dnf install -y python3-devel          # Python.h, needed by every C extension
source ~/venvs/torch-cpu/bin/activate
pip install pybind11 numpy certifi soundfile
# server only:
pip install fastapi uvicorn pydantic
```

Build (first time takes a few minutes; afterwards only changed files rebuild):

```sh
cd ~/gitprojects/kitten-tts-2-cpp
source ~/venvs/torch-cpu/bin/activate
cmake -S . -B build-py -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_KITTEN_TTS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DKITTEN_TTS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$(which python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-py --target _native -j 12
```

Install into the venv (run again after every rebuild):

```sh
pip install build-py/tools/kitten-tts/python
```

Check that the local build is the one imported:

```sh
cd /tmp && python -c "import kitten_tts_cpp, json, pathlib; print(kitten_tts_cpp.__file__); print(pathlib.Path(kitten_tts_cpp.__file__).with_name('_build_info.json').read_text())"
```

`"portable": false` means it is this build (the PyPI wheel says `true`).

Use:

```python
from kitten_tts_cpp import KittenTTS
m = KittenTTS(decoder="student_w4", threads=6, decoder_threads=4)
audio = m.generate("Hello there.", voice="Bruno")   # float32 numpy, 24000 Hz
```

## 4. Verification results

Machine: **AMD Ryzen 5 7600X** (Zen 4, 6 cores / 12 threads, one CCD, AVX-512), as reported by `lscpu` inside the toolbox. Note: the task description said Ryzen 9 7900X, but this toolbox runs on a 7600X. GCC 15.2.1, Python 3.13.13, torch 2.14.1+cpu, pybind11 3.1.0.

### 4.1 API contract

```
__init__ (self, model_name='KittenML/kitten-tts-2', cache_dir=None, device=None, hf_token=None, decoder=None, weights='packed', *, threads=8, decoder_threads=None, revision=None, offline=False, model_path=None, data_dir=None, repack=True, ca_file=None)
generate (self, text, voice=None, reference=None, reference_text=None, preset=None, max_new_tokens=1000, normalize=True, use_reference_prompt=None, temperature=None, top_k=None, top_p=None, min_p=None, advanced=None)
members ['advanced_options', 'available_decoders', 'available_voices', 'available_weights', 'close', 'generate', 'generate_stream', 'generate_to_file', 'normalize_text', 'sample_rate']
kitten_tts_cpp.__file__ = /home/BrockToller/venvs/torch-cpu/lib64/python3.13/site-packages/kitten_tts_cpp/__init__.py
```

Matches the contract exactly.

### 4.2 One load, three calls

`KittenTTS(decoder="student_w4", threads=6, decoder_threads=4)`, then `generate("One day, a little girl named Lily found a needle in her room.", voice="Bruno")` three times in one process. Load (import + LM + decoder): **1.39 s, once**.

| Call | Wall | Prompt (304 tok) | Gen tokens | Gen tok/s | Decoder | Audio | RTF |
|---|---|---|---|---|---|---|---|
| 1 | 5.29 s | 0.96 s | 121 | 45.9 | 1.39 s | 4.92 s | 1.06 |
| 2 | 5.30 s | 0.94 s | 117 | 45.7 | 1.51 s | 4.76 s | 1.10 |
| 3 | 4.43 s | 0.98 s | 117 | 45.8 | 0.62 s | 4.76 s | 0.91 |
| Native CLI (`build/bin/kitten-tts`, same settings) | | 1.03 s | 109 | 45.5 | 1.35 s | 4.44 s | 1.12 |
| Target from the task | | 0.94 s | | 45.7 | 1.34 s | | 1.08 |
| PyPI wheel (task, 5300G) | | 10.58 s | | 7.92 | 2.64 s | | 5.91 |

Gen tok/s is llama.cpp's own eval counter (same number the CLI prints). RTF = (LM + decoder time) / audio length; below 1 is faster than real time. The decoder gets faster after the first two calls because TorchScript optimizes the graph after warm-up runs. Token counts differ per call because sampling is random unless a seed is given.

### 4.3 Same output as the CLI

Same sentence, `seed=42` in both (`advanced={"seed": 42}` / `--seed 42`):

```
normalized equal: True
prompt ids equal: True
generated tokens equal: True 109 109
samples: (106560,) (106560,) max abs diff: 3.05e-05
```

The language model output is identical. The audio differs by at most 3e-5 (about -90 dB, inaudible), from floating-point summation order inside the multi-threaded torch decoder. WAVs for listening are in the session scratchpad: `py_seed42.wav`, `cli_seed42.wav`, `py_call1..3.wav`, `py_stream.wav`.

### 4.4 Streaming

`generate_stream()` on a 557-character, 5-sentence text yields 2 chunks (23.4 s and 16.7 s of audio; the first one after 20.7 s). Chunks are text pieces of up to about 380 characters (`chunk_chars`), so a text shorter than that yields one chunk. Closing the iterator early works: the chunk already being generated finishes, then it stops, and the model stays usable.

### 4.5 `ldd` (no CUDA)

```
$ readelf -d _native.cpython-313-x86_64-linux-gnu.so | grep NEEDED
 libgomp.so.1  libssl.so.3  libcrypto.so.3  libtorch.so  libtorch_cpu.so  libc10.so
 libstdc++.so.6  libm.so.6  libgcc_s.so.1  libc.so.6  ld-linux-x86-64.so.2

$ ldd _native.cpython-313-x86_64-linux-gnu.so
	libgomp.so.1 => .../torch-cpu/lib64/python3.13/site-packages/torch/lib/libgomp.so.1
	libssl.so.3 => /lib64/libssl.so.3
	libcrypto.so.3 => /lib64/libcrypto.so.3
	libtorch.so => .../site-packages/torch/lib/libtorch.so
	libtorch_cpu.so => .../site-packages/torch/lib/libtorch_cpu.so
	libc10.so => .../site-packages/torch/lib/libc10.so
	libstdc++.so.6 => /lib64/libstdc++.so.6
	libm.so.6 => /lib64/libm.so.6
	libgcc_s.so.1 => /lib64/libgcc_s.so.1
	libc.so.6 => /lib64/libc.so.6
	/lib64/ld-linux-x86-64.so.2
	libdl.so.2, libpthread.so.0, libz.so.1, librt.so.1 => /lib64/...
$ ldd ... | grep -iE "cuda|cudart|nvrtc|cublas"
(nothing)
```

No `libtorch_cuda`, no `libcudart`, and no `libllama`/`libggml` (those are linked in). A side finding: the PyPI `_native.so` also lists only `libtorch`, `libtorch_cpu` and `libc10`. Its CUDA stack comes in through the PyPI `torch==2.13.0` wheel it requires, which is the CUDA build.

Proof the CPU-specific code is in there: `objdump -d` finds 9162 AVX-512 (`zmm` register) instructions in this `_native.so`, and 0 in the PyPI one.

### 4.6 Server

From `~/patchedkitten/KittenTTS`:

```sh
KITTENML_BACKEND=cpp KITTENML_THREADS=6 python -m kittenml.kittentts2.docker.server --port 8001
```

```
GET  /health                              -> {"status":"ok","backend":"cpp","sample_rate":24000}
POST /v1/audio/speech  stream:false wav   -> HTTP 200, 444524 bytes, RIFF/WAVE, 24 kHz, 9.26 s audio, 9.9 s wall
POST /v1/audio/speech  stream:true  wav   -> HTTP 200, 411884 bytes, RIFF/WAVE, 24 kHz, 8.58 s audio, 9.5 s wall
GET  /v1/voices                           -> list of built-in voices
```

Built-in voices are tested. Voice cloning from raw audio is not supported by this package: the original `model.py` raises `NotImplementedError` for `reference=` (same as the PyPI wheel), so clones must be prepared offline with `tools/kitten-tts/prepare_reference.py`.

## 5. Building for a different CPU (Docker, another machine)

`-march=native` means "use every instruction this CPU has". On the 7600X that includes AVX-512. The compiler then puts AVX-512 instructions straight into the hot loops. If you copy that `.so` to a CPU without AVX-512 (for example a Zen 3 Ryzen 5000 like the 5300G, or many Intel desktop chips), the first time the CPU meets one of those instructions it stops the process with **SIGILL (illegal instruction)**. There is no fallback: the check happens in hardware, not in software.

So when you build an image on one machine to run on another, name the target CPU instead of `native`. ggml has its own options for this. With `GGML_NATIVE=OFF` it turns on SSE4.2/AVX/AVX2/FMA/F16C/BMI2 by default, and AVX-512 only if asked:

Zen 4 target (7600X/7900X; also builds fine on a non-Zen-4 build machine):

```sh
cmake ... -DGGML_NATIVE=OFF \
  -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON \
  -DCMAKE_C_FLAGS=-march=znver4 -DCMAKE_CXX_FLAGS=-march=znver4
```

Zen 3 / generic AVX2 (runs on almost any x86 CPU from the last 10 years, slower on Zen 4):

```sh
cmake ... -DGGML_NATIVE=OFF
```

(`-march=znver4` for the non-ggml code is optional; the speed comes from ggml's kernels. It also makes the whole `.so` Zen 4-only, so leave it out if unsure.) `_build_info.json` records `ggml_native` and `cxx_flags`, so you can check what an image was built with. ggml's `GGML_CPU_ALL_VARIANTS` (one binary that picks the best kernels at runtime) needs `GGML_BACKEND_DL` and shared libraries, so it does not fit this static single-`.so` build.

## 6. Notes for a Docker image

- **Torch**: install exactly `torch==2.14.1` from `https://download.pytorch.org/whl/cpu` before `pip install`ing this package (or rebuild against whatever torch the image has). The upstream Dockerfile installs a ROCm torch; the cpp backend does not need it.
- **Build deps** in the build stage: gcc/g++ with C++20, cmake, `python3-devel`, `openssl-devel`, `pip install pybind11`. **Runtime deps**: `libgomp` (comes with torch), `libssl.so.3`/`libcrypto.so.3` (OpenSSL 3, from the system; used only to download models from Hugging Face), numpy, soundfile (needs `libsndfile`), certifi.
- **Package size**: the `data/` directory (text normalizer grammars for all languages) is 138 MB and is needed. The `.so` is 8 MB.
- **Model cache**: models download on first use to `~/.cache/kitten-tts/<repo>/<revision>/`. Mount that path as a volume, or pre-download and pass `offline=True` / a local asset directory as `model_name`.
- **Threads**: the 7600X has 6 cores. `threads=6, decoder_threads=4` were best. Hyper-threads do not help. On a two-CCD chip (7900X), keeping the LM threads on one CCD (`taskset`/`--cpuset-cpus`) usually helps.
- **Server settings**: the patched `server.py` cpp backend passes only `threads=KITTENML_THREADS`. So it uses the repo's default decoder (not `student_w4`), and `decoder_threads` equals `threads`. To match the benchmark above, pass `decoder="student_w4", decoder_threads=4` there.
- **Wheel tag**: setuptools tags the installed package `py3-none-any` even though it contains a compiled `.so`. That is harmless for a local or Docker install, but do not upload such a wheel anywhere.
- **Logging**: llama.cpp prints load and context messages to stderr on every `generate` (same as the CLI). This is noise in server logs, not an error.
- **Global torch threads**: the decoder calls `at::set_num_threads(decoder_threads)`, which is process-wide. If something else in the same process uses torch, it shares that setting. This is existing CLI behavior, unchanged.
- **CMake warning** "runtime library [libgomp.so.1] ... may be hidden by files in .../torch/lib": both the CLI and `_native` load torch's bundled `libgomp`. That is the same OpenMP runtime family and is what the known-good CLI already does.
