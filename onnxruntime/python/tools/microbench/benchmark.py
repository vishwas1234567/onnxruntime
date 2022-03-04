from argparse import ArgumentParser
import time
import numpy
import onnxruntime as ort
import torch


def add_arguments(parser: ArgumentParser):
    parser.add_argument("--provider", required=False, type=str, default="rocm", help="Execution provider to use")
    parser.add_argument("--precision", required=False, type=str, default="fp16", help="Number format to use")
    parser.add_argument('--profiling', type=bool, default=False, help='If enable profiling')


def create_input_output_tensors(inputs, outputs):
    device = "cuda"
    input_tensors = {name: torch.from_numpy(array).to(device) for name, array in inputs.items()}
    output_tensors = {name: torch.from_numpy(array).to(device) for name, array in outputs.items()}
    return input_tensors, output_tensors


def numpy_type(torch_type):
    type_map = {torch.float32: numpy.float32,
                torch.float16: numpy.float16}
    return type_map[torch_type]


def create_io_binding(sess, input_tensors, output_tensors):
    io_binding = sess.io_binding()

    for name, tensor in input_tensors.items():
        io_binding.bind_input(name, tensor.device.type, 0, numpy_type(tensor.dtype), tensor.shape, tensor.data_ptr())

    for name, tensor in output_tensors.items():
        io_binding.bind_output(name, tensor.device.type, 0, numpy_type(tensor.dtype), tensor.shape, tensor.data_ptr())
  
    return io_binding


def create_session(onnx_file, args):
    sess_opt = ort.SessionOptions()
    sess_opt.enable_profiling = args.profiling
    if args.provider == "rocm":
        execution_provider = ["ROCMExecutionProvider"] 
    elif args.provider == "cuda":
        execution_provider = ["CUDAExecutionProvider"] 
    else:
        raise ValueError(f"The script doesn't support provider type '{args.provider}' yet.")

    sess = ort.InferenceSession(onnx_file, sess_options=sess_opt, providers=execution_provider)

    if args.provider == "rocm":
        assert 'ROCMExecutionProvider' in sess.get_providers()
    elif args.provider == "cuda":
        assert 'CUDAExecutionProvider' in sess.get_providers()

    return sess
 

def benchmark(onnx_file, inputs, outputs, args):
    sess = create_session(onnx_file, args)
    input_tensors, output_tensors = create_input_output_tensors(inputs, outputs)
    io_binding = create_io_binding(sess, input_tensors, output_tensors)

    # warm up    
    for iter in range(10):
      sess.run_with_iobinding(io_binding)    
    
    # measure 
    max_iters = 100
    start_time = time.time()
    for iter in range(max_iters):
        sess.run_with_iobinding(io_binding)    
    
    # time is in milliseconds
    elapsed_time = (time.time() - start_time) * 1000 / max_iters
    return elapsed_time