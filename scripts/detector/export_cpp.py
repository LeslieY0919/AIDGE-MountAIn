"""Exporte le détecteur MegaDetector v6 en C++ fp32 via AIDGE.

Chaîne, et pourquoi chaque étape existe :

1. **Coupe du graphe** avant le post-traitement YOLO. Le décodage DFL, la
   sigmoïde et le NMS ne sont pas exportables par AIDGE ; ils sont refaits en
   C++ (``cpp/detector_standalone/main.cpp``). On ne garde que les deux
   sorties ``Concat`` : ``boxes`` (1, 64, N) et ``scores`` (1, 3, N).
2. **Pliage des constantes par ONNX Runtime.** Les nœuds ``Constant`` deviennent
   des initializers, qu'AIDGE sait traiter.
3. **Promotion des scalaires 0-D.** L'ONNX d'Ultralytics contient des
   constantes de rang 0 que l'export C++ ne sait pas matérialiser.
4. **Export puis compilation CMake en ``-j1``.** La compilation interne
   d'AIDGE est parallèle et sature la RAM sur un graphe de cette taille.

Usage :
    python scripts/detector/export_cpp.py

Prérequis : ``models/detector/detector_v6_trained_640.onnx``, produit par
``scripts/detector/finetune.py``.
"""

from __future__ import annotations

import faulthandler
import shutil
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort

from mountain import config, paths
from mountain.aidge import export as aidge_export
from mountain.aidge import graph as aidge_graph

# Les gros graphes font parfois tomber AIDGE sur une erreur de segmentation :
# la trace native est la seule information exploitable.
faulthandler.enable()


def main() -> int:
    det = config.detector()
    in_size = det["model"]["input_size"]
    in_shape = [1, 3, in_size, in_size]

    raw = paths.DETECTOR_ONNX_TRAINED
    if not raw.exists():
        print(f"Modèle absent : {raw}", file=sys.stderr)
        print("Lancer d'abord `make detector-weights` puis `python scripts/detector/finetune.py`.",
              file=sys.stderr)
        return 1

    # 1. Coupe avant le post-traitement DFL/Sigmoid.
    cut = paths.DETECTOR_ONNX_CUT
    cut.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(
        str(raw),
        str(cut),
        input_names=det["graph_cut"]["input_names"],
        output_names=det["graph_cut"]["output_names"],
    )

    # 2. Pliage des constantes statiques par ONNX Runtime.
    folded = paths.DETECTOR_ONNX_CUT_FOLDED
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(cut), opts, providers=["CPUExecutionProvider"])

    # 3. Chargement AIDGE, avec promotion des constantes 0-D.
    graph = aidge_graph.load_graph(
        folded,
        in_shape,
        fold_constants=False,      # déjà fait par ONNX Runtime à l'étape 2
        remove_flatten=True,
        promote_scalars=True,
        allow_data_dependency=True,
    )

    # 4. Ne garder que les deux sorties Concat attendues par le C++.
    if not aidge_export.keep_concat_outputs(graph, expected=2):
        print("Avertissement : le graphe n'expose pas exactement 2 sorties Concat ; "
              "l'export conservera les sorties telles quelles.", file=sys.stderr)

    # 5. Export des sources C++.
    out_dir = paths.EXPORT_DETECTOR_FP32
    dummy = aidge_graph.make_tensor(np.zeros(in_shape, dtype=np.float32))
    aidge_export.export_cpp(graph, out_dir, [dummy], dtype="float32", compile_=False)

    # 6. Installer le harnais C++ et compiler. `-j1` pour préserver la RAM.
    shutil.copy(paths.CPP_DETECTOR_HARNESS, out_dir / "main.cpp")
    shutil.copy(paths.STB_HEADER, out_dir / "stb_image.h")

    build_dir = out_dir / "build"
    subprocess.run(
        ["cmake", "-B", str(build_dir), "-S", str(out_dir),
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS=-O3"],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "-j1"], check=True)

    print(f"Export et compilation terminés : {build_dir}")
    print(f"Essai : {build_dir}/run_export {paths.SAMPLE_IMAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
