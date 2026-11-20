"""Appel de ``aidge_export_cpp`` indépendamment de la version installée.

L'API d'export d'AIDGE a changé plusieurs fois : signature à mots-clés
variables (``model`` / ``graph_view`` / ``graphview``), puis passage à
``ExportConfig`` + ``aidge_core.export_utils.export``. Les scripts du projet
contenaient chacun leur propre introspection de signature, ce qui est
exactement le genre de code qu'on ne veut pas voir en cinq exemplaires.

``export_cpp`` encapsule les deux conventions : on tente d'abord l'API
``ExportConfig`` (AIDGE >= 0.10), on retombe sur l'ancienne sinon.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import aidge_core
import aidge_export_cpp


def _export_with_config(graph, out_dir: Path, input_tensors, dtype: str, compile_: bool):
    """Voie moderne : ``ExportConfig`` + ``aidge_core.export_utils.export``."""
    config = aidge_export_cpp.ExportConfig(
        export_folder=str(out_dir),
        dtype=dtype,
        cuda=False,
        compile=compile_,
    )
    aidge_core.export_utils.export(
        graph=graph,
        config=config,
        input_tensors=input_tensors,
    )


def _export_legacy(graph, out_dir: Path, input_tensors, scheduler):
    """Voie historique : ``aidge_export_cpp.export`` à signature variable."""
    params = inspect.signature(aidge_export_cpp.export).parameters
    kwargs: dict = {}

    if "export_folder_name" in params:
        kwargs["export_folder_name"] = str(out_dir)
    elif "export_folder" in params:
        kwargs["export_folder"] = str(out_dir)

    for name in ("model", "graph_view", "graphview"):
        if name in params:
            kwargs[name] = graph
            break

    if "scheduler" in params:
        kwargs["scheduler"] = scheduler
    if "inputs_tensor" in params and input_tensors:
        kwargs["inputs_tensor"] = input_tensors

    aidge_export_cpp.export(**kwargs)


def export_cpp(
    graph,
    out_dir: str | Path,
    input_tensors=None,
    *,
    dtype: str = "float32",
    compile_: bool = False,
):
    """Génère les sources C++ d'un graphe dans ``out_dir``.

    Args:
        graph: ``GraphView`` déjà préparé (dimensions propagées, backend posé).
        out_dir: dossier de sortie, créé si absent. Doit vivre sous
            ``exports/`` : son contenu est régénérable et non versionné.
        input_tensors: tenseurs d'entrée d'exemple, requis par certaines
            versions pour dimensionner les tampons générés.
        dtype: ``"float32"`` ou ``"int8"``.
        compile_: laisser ``False``. La compilation interne d'AIDGE est
            parallèle et sature la RAM sur les gros graphes ; on compile
            ensuite via CMake avec ``-j1``.

    Returns:
        Le ``Path`` du dossier d'export.
    """
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    scheduler = aidge_core.SequentialScheduler(graph)
    scheduler.generate_scheduling()

    try:
        _export_with_config(graph, out_dir, input_tensors or [], dtype, compile_)
    except (AttributeError, TypeError):
        _export_legacy(graph, out_dir, input_tensors or [], scheduler)

    return out_dir


def keep_concat_outputs(graph, expected: int = 2) -> bool:
    """Ne conserve que les sorties ``Concat`` du détecteur coupé.

    Après la coupe avant le post-traitement DFL/Sigmoid, le graphe garde des
    sorties parasites. Le pipeline C++ n'attend que les deux tenseurs
    ``boxes`` (1, 64, N) et ``scores`` (1, 3, N).

    Renvoie ``False`` si le compte ne correspond pas : dans ce cas on laisse
    le graphe intact plutôt que de produire un export silencieusement faux.
    """
    outputs = [
        (node, idx)
        for node, idx in graph.get_ordered_outputs()
        if idx == 0 and "Concat" in node.name()
    ]
    if len(outputs) != expected:
        return False
    graph.set_ordered_outputs(outputs)
    return True
