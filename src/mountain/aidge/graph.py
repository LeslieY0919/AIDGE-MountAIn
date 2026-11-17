"""Préparation d'un graphe AIDGE à partir d'un ONNX.

La séquence « charger, marquer les Producer comme constants, plier les
constantes, propager les dimensions, fixer le type et le backend » était
dupliquée à l'identique dans huit scripts. Chaque copie avait divergé sur un
détail (``remove_flatten`` ici, ``fuse_batchnorm`` là,
``allow_data_dependency`` ailleurs), ce qui rendait impossible de savoir
quelle variante était la bonne.

Tout est rassemblé ici, avec les options explicites. Les gardes
``hasattr`` sont conservées : l'API d'``aidge_core`` a bougé entre les
versions 0.9 et 0.10 et le projet doit tourner avec les deux.
"""

from __future__ import annotations

from pathlib import Path

import aidge_core
import aidge_onnx
import numpy as np

# ``aidge_backend_cpu`` n'est pas appelé directement mais son import
# enregistre les implémentations CPU des opérateurs. Sans lui,
# ``set_backend("cpu")`` échoue.
import aidge_backend_cpu  # noqa: F401


def mark_producers_constant(graph) -> None:
    """Marque tous les nœuds ``Producer`` comme constants.

    Prérequis au ``constant_folding`` : sans ce marquage, AIDGE considère les
    poids comme des entrées variables et ne plie rien.
    """
    for node in graph.get_nodes():
        if node.type() != "Producer":
            continue
        op = node.get_operator()
        if hasattr(op, "set_constant"):
            op.set_constant(True)
        elif hasattr(op, "attr") and hasattr(op.attr, "constant"):
            op.attr.constant = True


def promote_scalar_producers(graph, in_shape: list[int]) -> None:
    """Donne un rang non nul aux ``Producer`` scalaires (tenseurs 0-D).

    Certains ONNX exportés par Ultralytics contiennent des constantes 0-D que
    l'export C++ d'AIDGE ne sait pas matérialiser. On les reshape au rang de
    leur consommateur. Nécessite des dimensions déjà propagées.
    """
    for node in graph.get_nodes():
        if node.type() != "Producer":
            continue
        tensor = node.get_operator().get_output(0)
        if tensor is None or len(tensor.dims) != 0:
            continue

        target_rank = max(
            (
                len(child.get_operator().get_input(i).dims)
                for child in node.get_children()
                for i in range(child.get_nb_inputs())
                if child.get_operator().get_input(i) is not None
            ),
            default=0,
        )
        if target_rank == 0:
            continue

        promoted = aidge_core.Tensor(np.array(tensor).reshape((1,) * target_rank))
        promoted.set_datatype(tensor.dtype)
        promoted.set_data_format(
            aidge_core.dformat.nchw if target_rank == 4 else aidge_core.dformat.default
        )
        node.get_operator().set_output(0, promoted)


def fuse_batchnorm(graph) -> bool:
    """Fusionne les BatchNorm dans les convolutions amont, si disponible.

    Renvoie ``False`` quand la passe n'existe pas dans la version d'AIDGE
    installée ou qu'elle échoue — ce n'est pas bloquant, la fusion est une
    optimisation.
    """
    if not hasattr(aidge_core, "fuse_batchnorm"):
        return False
    try:
        aidge_core.fuse_batchnorm(graph)
        return True
    except Exception:
        return False


def load_graph(
    onnx_path: str | Path,
    in_shape: list[int],
    *,
    fold_constants: bool = True,
    remove_flatten: bool = False,
    fuse_bn: bool = False,
    promote_scalars: bool = False,
    allow_data_dependency: bool = False,
    datatype=None,
    backend: str = "cpu",
):
    """Charge un ONNX et renvoie un ``GraphView`` prêt à inférer ou exporter.

    Args:
        onnx_path: chemin du modèle ONNX.
        in_shape: forme d'entrée, p. ex. ``[1, 3, 320, 320]``.
        fold_constants: plier les constantes (recommandé).
        remove_flatten: remplacer les ``Flatten``, que l'export C++ gère mal.
        fuse_bn: tenter la fusion BatchNorm (utile avant une PTQ).
        promote_scalars: reshaper les constantes 0-D (ONNX YOLO d'Ultralytics).
        allow_data_dependency: autoriser la propagation à travers les
            ``Split``/``Reshape``/``Slice`` dont la forme dépend des données.
        datatype: type AIDGE cible, ``float32`` par défaut.
        backend: backend d'exécution.
    """
    if datatype is None:
        datatype = aidge_core.dtype.float32

    graph = aidge_onnx.load_onnx(str(onnx_path))
    mark_producers_constant(graph)

    if fold_constants:
        aidge_core.constant_folding(graph)
    if remove_flatten:
        aidge_core.remove_flatten(graph)
    if fuse_bn:
        fuse_batchnorm(graph)

    graph.set_datatype(datatype)
    graph.set_backend(backend)
    graph.forward_dims([in_shape], allow_data_dependency=allow_data_dependency)

    if promote_scalars:
        promote_scalar_producers(graph, in_shape)
        # Les dimensions doivent être repropagées après le reshape.
        graph.forward_dims([in_shape], allow_data_dependency=allow_data_dependency)

    return graph


def make_tensor(array: np.ndarray, *, nchw: bool = True):
    """numpy -> ``aidge_core.Tensor`` float32, format mémoire explicite."""
    tensor = aidge_core.Tensor(np.ascontiguousarray(array, dtype=np.float32))
    tensor.set_datatype(aidge_core.dtype.float32)
    if nchw and len(array.shape) == 4:
        tensor.set_data_format(aidge_core.dformat.nchw)
    return tensor


def run_forward(graph, input_array: np.ndarray) -> list[np.ndarray]:
    """Inférence CPU sur un graphe préparé. Renvoie les sorties ordonnées."""
    scheduler = aidge_core.SequentialScheduler(graph)
    scheduler.forward(data=[make_tensor(input_array)])
    return [
        np.array(node.get_operator().get_output(idx))
        for node, idx in graph.get_ordered_outputs()
    ]


def summarize(graph) -> dict:
    """Compte les nœuds et repère les opérateurs non natifs (``Generic``).

    Un opérateur ``Generic`` signale une couche qu'AIDGE n'implémente pas :
    c'est exactement ce qui fait échouer l'export C++. À vérifier avant toute
    tentative d'export.
    """
    nodes = list(graph.get_nodes())
    generic = [
        n.type()
        for n in nodes
        if "Generic" in type(n.get_operator()).__name__ or "Generic" in n.type()
    ]
    return {
        "nb_nodes": len(nodes),
        "op_types": sorted({n.type() for n in nodes}),
        "generic_ops": generic,
    }
