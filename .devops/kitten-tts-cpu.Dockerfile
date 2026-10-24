# KittenTTS2 OpenAI-compatible server on the native C++ CPU backend (kitten_tts_cpp).
#
# Build from the repo root; the patched KittenTTS checkout is passed as a second build context:
#   podman build -f .devops/kitten-tts-cpu.Dockerfile \
#     --build-context kittenml=$HOME/patchedkitten/KittenTTS \
#     --build-arg KITTEN_CPU=native -t kitten-tts-cpu .
#
# KITTEN_CPU: native = the build machine's CPU (crashes with SIGILL on CPUs that lack its instructions)
#             znver4 = Zen 4 and newer (Ryzen 7000, Ryzen 9000, EPYC 9004), uses AVX-512
#             avx2   = portable x86-64 with AVX2 (Zen 1-3, most Intel since 2013), slower on Zen 4

ARG FEDORA_VERSION=42
ARG TORCH_VERSION=2.14.1

FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION} AS build
ARG TORCH_VERSION
ARG KITTEN_CPU=native

RUN dnf install -y --setopt=install_weak_deps=False gcc-c++ cmake python3-devel openssl-devel \
    && dnf clean all

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir pybind11==3.1.0 numpy==2.5.4

WORKDIR /src
COPY . .
RUN test -f vendor/kitten-text-processing/CMakeLists.txt || { echo "vendor/kitten-text-processing is empty: run git submodule update --init" >&2; exit 1; }

RUN case "$KITTEN_CPU" in \
        native) CPU_FLAGS="" ;; \
        znver4) CPU_FLAGS="-DGGML_NATIVE=OFF -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON" ;; \
        avx2)   CPU_FLAGS="-DGGML_NATIVE=OFF" ;; \
        *) echo "unknown KITTEN_CPU=$KITTEN_CPU (native|znver4|avx2)" >&2; exit 1 ;; \
    esac \
    && cmake -S . -B build-py -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DGGML_CUDA=OFF -DLLAMA_BUILD_KITTEN_TTS=ON \
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DKITTEN_TTS_BUILD_PYTHON=ON -DPython_EXECUTABLE=/opt/venv/bin/python \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
        -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
        $CPU_FLAGS \
    && cmake --build build-py --target _native -j "$(nproc)" \
    && pip wheel --no-cache-dir --no-deps build-py/tools/kitten-tts/python -w /wheels \
    && cat build-py/tools/kitten-tts/python/kitten_tts_cpp/_build_info.json


FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION}
ARG TORCH_VERSION

RUN dnf install -y --setopt=install_weak_deps=False python3 libsndfile \
    && dnf clean all

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy==2.5.4 soundfile==0.14.0 certifi \
        fastapi==0.162.0 uvicorn==0.54.0 pydantic==2.14.1

COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl && rm -rf /wheels

# Only the server code is used; installing the kittenml package would pull its PyTorch model dependencies.
COPY --from=kittenml kittenml /opt/kittenml/kittenml

ENV PYTHONPATH=/opt/kittenml \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/cache \
    KITTENML_VOICES_DIR=/cache/voices \
    KITTENML_BACKEND=cpp \
    KITTENML_THREADS=6 \
    KITTENML_DECODER=student_w4 \
    KITTENML_DECODER_THREADS=4

# models download to /cache/kitten-tts on first start; mount it to keep them
VOLUME /cache
EXPOSE 8001
WORKDIR /workspace
CMD ["python", "-m", "kittenml.kittentts2.docker.server", "--host", "0.0.0.0", "--port", "8001"]
