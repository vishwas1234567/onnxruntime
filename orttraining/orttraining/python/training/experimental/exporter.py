import torch
import torch.onnx.utils
import torch.onnx.symbolic_helper

def _export(graph, operator_export_type, _disable_torch_constant_prop=False, fixed_batch_size=False,
                    params_dict=None, dynamic_axes=None, input_names=None, module=None):
    torch.onnx.symbolic_helper._set_onnx_shape_inference(True)
    torch._C._jit_pass_inline(graph)

    # Remove fork/wait nodes
    torch._C._jit_pass_inline_fork_wait(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_lower_all_tuples(graph)

    # we now record some ops like ones/zeros
    # into a trace where we previously recorded constants.
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
    if _disable_torch_constant_prop is False:
        torch._C._jit_pass_constant_propagation(graph)

    torch.onnx.utils._split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_canonicalize_graph_fuser_ops(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_fuse_addmm(graph)
    torch._C._jit_pass_lint(graph)
    from torch.onnx.symbolic_helper import _onnx_shape_inference, _set_opset_version, _set_operator_export_type

    _set_opset_version(12)
    _set_operator_export_type(torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    from torch.onnx.symbolic_helper import _export_onnx_opset_version

    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lower_all_tuples(graph)
    # in _jit_pass_onnx, symbolic functions are called for each node for conversion.
    # However, there are nodes that cannot be converted without additional context.
    # For example, the number of outputs from split (and whether it is static or dynamic) is unknown
    # until the point where it is unpacked by listUnpack node.
    # This pass does a preprocess, and prepares the nodes such that enough context can be received
    # by the symbolic function.
    torch._C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)
    torch._C._jit_pass_onnx_preprocess(graph)

    # onnx does not support tuples, so try to remove them
    torch._C._jit_pass_lint(graph)

    # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
    torch._C._jit_pass_prepare_division_for_onnx(graph)

    torch._C._jit_pass_onnx_remove_print(graph)
    torch._C._jit_pass_onnx_preprocess_caffe2(graph)

    if operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        torch.onnx.symbolic_helper._quantized_ops.clear()
        # Unpack quantized weights for conv and linear ops and insert into graph.
        torch._C._jit_pass_onnx_unpack_quantized_weights(graph, {})
        # Insert permutes before and after each conv op to ensure correct order.
        torch._C._jit_pass_onnx_quantization_insert_permutes(graph, {})

        # Find consecutive permutes that are no-ops and remove them.
        torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
        graph(%Pi):
            %Pq = quantized::nhwc2nchw(%Pi)
            %Pr = quantized::nchw2nhwc(%Pq)
            return (%Pr)""", """
        graph(%Ri):
            return (%Ri)""", graph)

    # onnx only supports tensors, so we turn all out number types into tensors
    torch._C._jit_pass_erase_number_types(graph)

    if _onnx_shape_inference:
        input_names = [] if input_names is None else input_names
        dynamic_axes = {} if dynamic_axes is None else dynamic_axes
        torch._C._jit_pass_onnx_set_dynamic_input_shape(graph, dynamic_axes, input_names)
    torch._C._jit_pass_onnx_lint(graph)
    graph = torch._C._jit_pass_onnx(graph, operator_export_type)
    torch._C._jit_pass_onnx_lint(graph)
    torch._C._jit_pass_lint(graph)

    #print('[utils.py] _jit_pass_onnx_scalar_type_analysis\n', graph)
    torch._C._jit_pass_onnx_scalar_type_analysis(graph, True, _export_onnx_opset_version)
    #print('[utils.py] _jit_pass_onnx_scalar_type_analysis done\n', graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_onnx_peephole(graph, _export_onnx_opset_version, fixed_batch_size)
    torch._C._jit_pass_lint(graph)

    # graph is not a valid jit graph anymore because types have been replaced
    # (e.g. int with Tensor), so it now contains operators that don't actually
    # exist. We can't run normal dead code elimination because it'd fail trying
    # to look up if an operator has side effects, but we can run a dead code
    # elimination variant that doesn't need to look up if an op has side effects.
    torch._C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
    if _onnx_shape_inference:
        torch._C._jit_pass_onnx_graph_shape_type_inference(graph, {}, _export_onnx_opset_version)

    import random, string, tempfile
    seed  = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    proto, export_map, val_use_external_data_format = graph._export_onnx({}, 12, {}, False, operator_export_type, False, False, {}, True, f'/bert_ort/wechi/model_{seed}', {})
    assert len(export_map) == 0
    model_location = f'{tempfile.gettempdir()}/model_{seed}.onnx'
    with torch.serialization._open_file_like(model_location, "wb") as opened_file:
        opened_file.write(proto)
    return model_location