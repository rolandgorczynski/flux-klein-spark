FROM nvcr.io/nvidia/pytorch:25.09-py3
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
RUN python -m venv --system-site-packages /opt/flux
ENV PATH="/opt/flux/bin:${PATH}" PYTHONUNBUFFERED=1 HF_HOME=/data/hf-cache GRADIO_TEMP_DIR=/data/output GRADIO_ANALYTICS_ENABLED=False DO_NOT_TRACK=1 NO_ALBUMENTATIONS_UPDATE=1
COPY requirements.txt ./
RUN PIP_CONSTRAINT= python -m pip install --no-cache-dir -r requirements.txt
# Old optional torchao in the NVIDIA image conflicts with current Transformers.
# BF16 inference does not need a quantization package.
RUN /usr/bin/python3 -m pip uninstall -y torchao
COPY *.py ./
EXPOSE 7861
ENTRYPOINT ["python", "app.py"]
