"""
Graphviz-based visualization utility for the mini_ml autograd engine.

This module generates computation graphs showing:

- Tensor values
- Gradients
- Operations
- Parent-child dependencies

Example
-------
>>> dot = draw_dot(output)
>>> dot.render("autograd_graph")
"""

import os
import sys
import platform
from pathlib import Path

if platform.system() == "Windows":
    os.environ["PATH"] = (
        r"C:\Program Files\Graphviz\bin;"
        + os.environ["PATH"]
    )

import graphviz
import numpy as np

repo_root = Path(__file__).resolve().parent.parent

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from mini_ml.nn.autograd import Value


def get_all_nodes_and_edges(root_node: Value):
    """
    Traverse computation graph and collect all nodes and edges.
    """

    nodes = set()
    edges = set()
    visited = set()

    def build(node):

        node_id = id(node)

        if node_id in visited:
            return

        visited.add(node_id)
        nodes.add(node)

        for parent in getattr(node, "_prev", ()):

            edges.add((parent, node))
            build(parent)

    build(root_node)

    return nodes, edges


def format_tensor(x, max_elements=12):
    """
    Format tensors for compact display.
    """

    arr = np.asarray(x)

    if arr.ndim == 0:
        return f"{float(arr):.6f}"

    if arr.size <= max_elements:

        return np.array2string(
            arr,
            precision=4,
            suppress_small=True,
            separator=", ",
        )

    return f"shape={arr.shape}"


def build_node_label(node):
    """
    Create Graphviz record label.
    """

    title = getattr(node, "label", "")

    if not title:
        title = "Value"

    data_str = format_tensor(node.data)
    grad_str = format_tensor(node.grad)

    return (
        "{"
        f"{title}"
        "|"
        f"data={data_str}"
        "|"
        f"grad={grad_str}"
        "}"
    )


def draw_dot(
    root_node: Value,
    format="png",
    rankdir="TB",
):
    """
    Build Graphviz representation of a computation graph.

    Parameters
    ----------
    root_node : Value
        Final output node.

    format : str
        Graph output format.

    rankdir : {"LR", "TB"}
        Layout direction.
    """

    if rankdir not in {"LR", "TB"}:
        raise ValueError(
            "rankdir must be 'LR' or 'TB'."
        )

    nodes, edges = get_all_nodes_and_edges(
        root_node
    )

    dot = graphviz.Digraph(
        format=format,
        graph_attr={
            "rankdir": rankdir,
            "splines": "spline",
            "nodesep": "0.45",
            "ranksep": "0.7",
            "pad": "0.2",
            "fontsize": "16",
            "label": "Autograd Computation Graph",
            "labelloc": "t",
        },
    )

    # Tensor nodes

    for node in sorted(
        nodes,
        key=lambda n: id(n),
    ):

        uid = str(id(node))

        dot.node(
            uid,
            label=build_node_label(node),
            shape="record",
            style="rounded,filled",
            fillcolor="#FFF8DC",
            color="#444444",
            fontname="Consolas",
        )

        op = getattr(node, "_op", "")

        if op:

            op_uid = f"{uid}_{op}"

            dot.node(
                op_uid,
                label=op,
                shape="circle",
                style="filled",
                fillcolor="#87CEFA",
                color="#444444",
                fontname="Consolas",
            )

            dot.edge(
                op_uid,
                uid,
            )

    # Parent -> operation

    for parent, child in sorted(
        edges,
        key=lambda e: (id(e[0]), id(e[1])),
    ):

        op = getattr(child, "_op", "")

        if op:

            dot.edge(
                str(id(parent)),
                f"{id(child)}_{op}",
            )

    return dot


if __name__ == "__main__":

    from mini_ml.nn.modules import Sequential
    from mini_ml.nn.modules.activations import ReLU
    from mini_ml.nn.modules.linear import Linear

    print("\n--- Visualization Example ---")

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )

    try:
        model.layers[0].weight.label = "W1"
        model.layers[0].bias.label = "b1"

        model.layers[2].weight.label = "W2"
        model.layers[2].bias.label = "b2"

    except Exception:
        pass

    x = Value(
        np.array([[1.5, -2.0]]),
        label="x",
    )

    output = model(x)
    output.label = "output"

    dot_graph = draw_dot(
        output,
        format="png",
        rankdir="TB",
    )

    try:

        output_file = dot_graph.render(
            "assets/autograd_computation_graph",
            cleanup=True,
            view=False,
        )

        print(
            f"Graph saved to: {output_file}"
        )

    except graphviz.backend.execute.ExecutableNotFound:

        print(
            "\nGraphviz executable not found.\n"
            "Install Graphviz and add its bin directory to PATH."
        )

    except Exception as exc:

        print(
            f"\nRendering failed:\n{exc}"
        )
