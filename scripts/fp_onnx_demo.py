"""Generate tiny executable ONNX test models; these are NOT trained inspection models."""
from pathlib import Path


def make_demo_model(path, *, detection=False, gain=1.0, bias=0.0):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    constants = [numpy_helper.from_array(np.array([gain], np.float32), "gain"),
                 numpy_helper.from_array(np.array([bias], np.float32), "bias"),
                 numpy_helper.from_array(np.array([0], np.float32), "zero"),
                 numpy_helper.from_array(np.array([1], np.float32), "one")]
    nodes = [helper.make_node("ReduceMean", ["images"], ["mean"], axes=[1, 2, 3], keepdims=0),
             helper.make_node("Mul", ["mean", "gain"], ["scaled"]),
             helper.make_node("Add", ["scaled", "bias"], ["shifted"]),
             helper.make_node("Clip", ["shifted", "zero", "one"], ["score"])]
    if detection:
        constants.extend([numpy_helper.from_array(np.array([4, 4, 28, 28], np.float32), "box"),
                          numpy_helper.from_array(np.array([1, 1, 6], np.int64), "out_shape")])
        nodes.extend([helper.make_node("Concat", ["box", "score", "zero"], ["row"], axis=0),
                      helper.make_node("Reshape", ["row", "out_shape"], ["output"])])
        shape = [1, 1, 6]
    else:
        nodes.extend([helper.make_node("Sub", ["one", "score"], ["normal"]),
                      helper.make_node("Concat", ["normal", "score"], ["row"], axis=0),
                      helper.make_node("Unsqueeze", ["row", "axes"], ["output"])])
        constants.append(numpy_helper.from_array(np.array([0], np.int64), "axes"))
        shape = [1, 2]
    graph = helper.make_graph(nodes, "LV executable adapter test ONLY",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--detection", action="store_true")
    args = parser.parse_args()
    print(make_demo_model(args.output, detection=args.detection))
