# Environnement Aidge (Linux x86_64) : Aidge 0.10.1 + aidge_tosa + aidge_export_arm_ethos + Vela 5.1.0, pour exporter
# les modèles et prévisualiser le partitionnement NPU. Aidge ne publie des wheels que pour Linux x86_64 et Windows :
# sous macOS, il tourne donc dans Docker sur une machine Linux ; construction et exécution via scripts/dev2_aidge.sh.
# Uniquement des wheels précompilées, sans compilateur ; environ 150 Mo à télécharger. Le firmware de la carte est
# compilé et flashé à part (scripts/build_img_class.sh).
FROM python:3.12-slim-bookworm

ARG AIDGE_VERSION=0.10.1
ARG TOSA_TOOLS_VERSION=2026.8.0
ARG AIDGE_TOSA_COMMIT=d0308c29ddeecac7baa1fe9917ef38e664804412
ARG AIDGE_EXPORT_ETHOS_COMMIT=da083686ecb94119d06b2c39b34ee4dc583cb8d5
# Si le réseau est lent, on peut passer par un miroir, par exemple --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_INDEX_URL=${PIP_INDEX_URL}

RUN pip install aidge-core==${AIDGE_VERSION} aidge-backend-cpu==${AIDGE_VERSION} aidge-onnx==${AIDGE_VERSION} \
        aidge-quantization==${AIDGE_VERSION} aidge-export-cpp==${AIDGE_VERSION} \
        aidge-export-arm-cortexm==${AIDGE_VERSION} aidge-benchmark==${AIDGE_VERSION} ethos-u-vela==5.1.0 pillow

# Bibliothèque de sérialisation et modèle de référence TOSA : tosa-tools précompilé depuis PyPI, à la place de
# third_party/tosa_install.sh d'aidge_tosa (qui compile le C++ depuis les sources). Les modules fournis (serializer,
# tosa, tosa_reference_model) sont ceux qu'importe aidge_tosa.
RUN pip install tosa-tools==${TOSA_TOOLS_VERSION}

# aidge_tosa et aidge_export_arm_ethos ne sont pas sur PyPI : installés depuis GitLab à des commits figés.
# aidge_export_arm_ethos sans dépendances : cmake/ninja ne servent qu'à compiler le firmware, ce qui se fait hors du
# conteneur ; Vela est déjà installé plus haut.
RUN pip install "https://gitlab.eclipse.org/eclipse/aidge/aidge_tosa/-/archive/${AIDGE_TOSA_COMMIT}/aidge_tosa.tar.gz" \
 && pip install --no-deps "https://gitlab.eclipse.org/eclipse/aidge/aidge_export_ethos/-/archive/${AIDGE_EXPORT_ETHOS_COMMIT}/aidge_export_ethos.tar.gz"

WORKDIR /work
