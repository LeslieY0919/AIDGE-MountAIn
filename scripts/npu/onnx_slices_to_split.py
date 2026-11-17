#!/usr/bin/env python3
# Remplace, dans un ONNX, chaque groupe de Slice qui découpe un même tenseur en morceaux contigus le long des canaux
# (axe 1, pas de 1, du canal 0 au dernier) par un Split équivalent. Le calcul ne change pas.
#
# Pourquoi : selon l'export, les C2f de YOLOv10 séparent leurs canaux par deux Slice ou par un Split. Le fork
# aidge-tosa-fork exporte le graphe en NHWC ; il y convertit l'axe d'un Split, mais pas celui d'un Slice, qui
# découpe alors la hauteur et l'export s'arrête (« Graph dimensions could not be forwarded after transpose
# sinking »). Voir notes/2026-11-17-aidge-vers-npu.md.
#
# Ne demande que le paquet onnx :
#   python scripts/npu/onnx_slices_to_split.py models/yolov10n.onnx models/yolov10n_split.onnx
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def constant_values(graph):
    """Valeurs connues à l'avance : initialiseurs et sorties des nœuds Constant."""
    values = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    values[node.output[0]] = numpy_helper.to_array(attr.t)
    return values


def channel_counts(model):
    """Nombre de canaux (dimension 1) des tenseurs de rang 4, d'après l'inférence de formes d'onnx."""
    inferred = onnx.shape_inference.infer_shapes(model).graph
    counts = {}
    for vi in list(inferred.value_info) + list(inferred.input) + list(inferred.output):
        dims = vi.type.tensor_type.shape.dim
        if len(dims) == 4 and dims[1].HasField("dim_value"):
            counts[vi.name] = dims[1].dim_value
    return counts


def slices_to_split(model):
    """Modifie le modèle en place ; renvoie le nombre de Slice remplacés."""
    graph = model.graph
    values = constant_values(graph)
    channels = channel_counts(model)

    # Slice par tenseur découpé : (début, fin, nœud), seulement sur l'axe 1 avec un pas de 1
    groups = defaultdict(list)
    for node in graph.node:
        if node.op_type != "Slice" or len(node.input) not in (4, 5):
            continue
        starts, ends, axes = (values.get(name) for name in node.input[1:4])
        steps = values.get(node.input[4]) if len(node.input) == 5 else np.array([1])
        if any(v is None for v in (starts, ends, axes, steps)) or list(axes) != [1] or list(steps) != [1]:
            continue
        groups[node.input[0]].append((int(starts[0]), int(ends[0]), node))

    replacement, removed = {}, set()
    for tensor, slices in groups.items():
        slices.sort(key=lambda s: s[0])
        contiguous = slices[0][0] == 0 and all(a[1] == b[0] for a, b in zip(slices, slices[1:]))
        if len(slices) < 2 or not contiguous or slices[-1][1] != channels.get(tensor):
            print(f"laissé tel quel : {tensor} {[(s, e) for s, e, _ in slices]} ({channels.get(tensor)} canaux)")
            continue
        sizes = [e - s for s, e, _ in slices]
        sizes_name = f"{tensor}_split_sizes"
        graph.initializer.append(numpy_helper.from_array(np.array(sizes, np.int64), sizes_name))
        # Le Split prend la place du premier Slice : son entrée est déjà calculée, ses sorties pas encore lues
        replacement[id(slices[0][2])] = helper.make_node(
            "Split", [tensor, sizes_name], [n.output[0] for _, _, n in slices], axis=1,
            name=f"{slices[0][2].name}_as_split")
        removed.update(id(n) for _, _, n in slices)
        print(f"{tensor} : {len(slices)} Slice -> Split {sizes}")

    nodes = []
    for node in graph.node:
        if id(node) in replacement:
            nodes.append(replacement[id(node)])
        elif id(node) not in removed:
            kept = onnx.NodeProto()
            kept.CopyFrom(node)  # copie : le nœud survit au vidage de graph.node quelle que soit l'implémentation
            nodes.append(kept)
    del graph.node[:]
    graph.node.extend(nodes)
    return len(removed)


def main():
    ap = argparse.ArgumentParser(description="Slice des canaux -> Split équivalent, dans un ONNX")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    model = onnx.load(str(args.src))
    count = slices_to_split(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(args.dst))
    print(f"{args.dst} : {count} Slice remplacés")


if __name__ == "__main__":
    main()
