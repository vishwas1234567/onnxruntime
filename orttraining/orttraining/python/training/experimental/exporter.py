import torch
import torch.onnx.utils
import torch.onnx.symbolic_helper

# Only this version is tested.
onnx_version = 12

def _export(graph, operator_export_type):
    torch.onnx.symbolic_helper._set_onnx_shape_inference(True)
    torch.onnx.symbolic_helper._set_opset_version(onnx_version)
    torch.onnx.symbolic_helper._set_operator_export_type(
      torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    graph = torch.onnx.utils._optimize_graph(graph, operator_export_type, params_dict={})

    import tempfile
    model_location = f'{tempfile.gettempdir()}/model_{tempfile.gettempprefix()}.onnx'
    proto, export_map, val_use_external_data_format = graph._export_onnx(
      {}, onnx_version, {}, False,
      operator_export_type, False, False,
      {}, True, '', {})
    with torch.serialization._open_file_like(model_location, "wb") as opened_file:
        opened_file.write(proto)
    return model_location
