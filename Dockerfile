# Oracle runtime image: the package plus its trained models, for hierarchical
# classification of transient/variable alerts. CPU build to stay lean; for GPU,
# swap to an nvidia/cuda base and drop the CPU torch index below.
FROM python:3.12-slim

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /opt/oracle

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /opt/oracle

# CPU torch keeps the image lean (the default CUDA wheel pulls ~2 GB of
# libraries); the pinned versions match pyproject, so `pip install .` reuses them.
RUN pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install .

# Fail the build if the package or its heavy deps don't import.
RUN python -c "import oracle, oracle.architectures, oracle.taxonomies; import torch, transformers, timm; print('oracle ok')"

CMD ["python"]
