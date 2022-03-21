# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from pathlib import Path
from typing import List
import sys
import os
import logging
import numpy
import torch
from transformers import T5Config
from onnxruntime import InferenceSession
from t5_encoder import T5Encoder, T5EncoderInputs
from t5_decoder import T5DecoderInit
from past_helper import PastKeyValuesHelper

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from torch_onnx_export_helper import torch_onnx_export

logger = logging.getLogger(__name__)


class T5EncoderDecoderInit(torch.nn.Module):
    """ A combination of T5Encoder and T5DecoderInit.
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 lm_head: torch.nn.Module,
                 config: T5Config,
                 decoder_start_token_id: int = None):
        super().__init__()
        self.config = config
        self.t5_encoder = T5Encoder(encoder, config)
        self.t5_decoder_init = T5DecoderInit(decoder, lm_head, config, decoder_start_token_id)

    def forward(self,
                encoder_input_ids: torch.Tensor,
                encoder_attention_mask: torch.Tensor,
                decoder_input_ids: torch.Tensor = None):
        encoder_hidden_states: torch.FloatTensor = self.t5_encoder(encoder_input_ids, encoder_attention_mask)
        lm_logits, past_self, past_cross = self.t5_decoder_init(decoder_input_ids, encoder_attention_mask,
                                                                encoder_hidden_states)
        return lm_logits, encoder_hidden_states, past_self, past_cross


class T5EncoderDecoderInitInputs:
    def __init__(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids=None):
        self.encoder_input_ids: torch.LongTensor = encoder_input_ids
        self.encoder_attention_mask: torch.LongTensor = encoder_attention_mask
        self.decoder_input_ids: torch.LongTensor = decoder_input_ids

    @staticmethod
    def create_dummy(config: T5Config, batch_size: int, encode_sequence_length: int, use_decoder_input_ids: int,
                     device: torch.device):  # -> T5EncoderDecoderInitInputs:
        encoder_inputs: T5EncoderInputs = T5EncoderInputs.create_dummy(batch_size, encode_sequence_length,
                                                                       config.vocab_size, device)
        decoder_input_ids = None
        if use_decoder_input_ids:
            decoder_input_ids = torch.ones(
                (batch_size, 1), dtype=torch.long, device=device) * config.decoder_start_token_id

        return T5EncoderDecoderInitInputs(encoder_inputs.input_ids, encoder_inputs.attention_mask, decoder_input_ids)

    def to_list(self) -> List:
        input_list = [self.encoder_input_ids, self.encoder_attention_mask]
        if self.decoder_input_ids is not None:
            input_list.append(self.decoder_input_ids)
        return input_list


class T5EncoderDecoderInitHelper:
    @staticmethod
    def export_onnx(model: T5EncoderDecoderInit,
                    device: torch.device,
                    onnx_model_path: str,
                    use_decoder_input_ids: bool = True,
                    verbose: bool = True,
                    use_external_data_format: bool = False):
        """Export decoder to ONNX

        Args:
            model (T5EncoderDecoderInit): the model to export
            device (torch.device): device of decoder object
            onnx_model_path (str): onnx path
            verbose (bool, optional): print verbose information. Defaults to True.
            use_external_data_format (bool, optional): use external data format or not. Defaults to False.
        """
        assert isinstance(model, T5EncoderDecoderInit)

        inputs = T5EncoderDecoderInitInputs.create_dummy(model.config,
                                                         batch_size=2,
                                                         encode_sequence_length=3,
                                                         use_decoder_input_ids=use_decoder_input_ids,
                                                         device=device)
        input_list = inputs.to_list()
        outputs = model(*input_list)

        present_names = PastKeyValuesHelper.get_past_names(model.config.num_layers, present=True)

        output_names = ["logits", "encoder_hidden_states"] + present_names

        # Shape of input tensors (sequence_length==1):
        #    input_ids: (batch_size, sequence_length)
        #    encoder_attention_mask: (batch_size, encode_sequence_length)
        #    encoder_hidden_states: (batch_size, encode_sequence_length, hidden_size)
        #    past_self_*: (batch_size, num_heads, past_decode_sequence_length, hidden_size/num_heads)
        #    past_cross_*: (batch_size, num_heads, encode_sequence_length, hidden_size/num_heads)

        # Shape of output tensors:
        #    logits: (batch_size, sequence_length, vocab_size)
        #    past_self_*: (batch_size, num_heads, past_decode_sequence_length + sequence_length, hidden_size/num_heads)
        #    past_cross_*: (batch_size, num_heads, encode_sequence_length, hidden_size/num_heads)

        input_names = ["encoder_input_ids", "encoder_attention_mask"]

        # ONNX exporter might mark dimension like 'Transposepresent_value_self_1_dim_2'. Use more friendly string here.
        sequence_length = '1'
        num_heads = str(model.config.num_heads)
        hidden_size = str(model.config.d_model)
        head_size = str(model.config.d_model // model.config.num_heads)

        dynamic_axes = {
            'encoder_input_ids': {
                0: 'batch_size',
                1: 'encode_sequence_length'
            },
            'encoder_attention_mask': {
                0: 'batch_size',
                1: 'encode_sequence_length'
            },
            'encoder_hidden_states': {
                0: 'batch_size',
                1: 'encode_sequence_length',
                2: hidden_size
            },
            "logits": {
                0: 'batch_size',
                1: sequence_length
            }
        }

        if use_decoder_input_ids:
            input_names.append("decoder_input_ids")
            dynamic_axes["decoder_input_ids"] = {0: 'batch_size', 1: sequence_length}

        for name in present_names:
            if "cross" in name:
                dynamic_axes[name] = {0: 'batch_size', 1: num_heads, 2: 'encode_sequence_length', 3: head_size}

            else:  # self attention past state
                dynamic_axes[name] = {0: 'batch_size', 1: num_heads, 2: sequence_length, 3: head_size}

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
        torch_onnx_export(model,
                          args=tuple(input_list),
                          f=onnx_model_path,
                          export_params=True,
                          input_names=input_names,
                          output_names=output_names,
                          example_outputs=outputs,
                          dynamic_axes=dynamic_axes,
                          opset_version=12,
                          do_constant_folding=True,
                          use_external_data_format=use_external_data_format,
                          verbose=verbose)

    @staticmethod
    def onnxruntime_inference(ort_session, inputs: T5EncoderDecoderInitInputs):
        """ Run inference of ONNX model.
        """
        logger.debug(f"start onnxruntime_inference")

        ort_inputs = {
            'encoder_input_ids': numpy.ascontiguousarray(inputs.encoder_input_ids.cpu().numpy()),
            'encoder_attention_mask': numpy.ascontiguousarray(inputs.encoder_attention_mask.cpu().numpy()),
        }
        if inputs.decoder_input_ids is not None:
            ort_inputs['decoder_input_ids'] = numpy.ascontiguousarray(inputs.decoder_input_ids.cpu().numpy())

        ort_outputs = ort_session.run(None, ort_inputs)
        return ort_outputs

    @staticmethod
    def verify_onnx(model: T5EncoderDecoderInit, ort_session: InferenceSession, device: torch.device, max_cases=4):
        """ Compare the result from PyTorch and OnnxRuntime to verify the ONNX model is good.
        """
        ort_inputs = ort_session.get_inputs()
        use_decoder_input_ids = len(ort_inputs) == 3

        test_cases = [(4, 11), (1, 2), (3, 1), (8, 5)]
        test_cases_max_diff = []
        for (batch_size, encode_sequence_length) in test_cases[:max_cases]:
            inputs = T5EncoderDecoderInitInputs.create_dummy(model.config,
                                                             batch_size,
                                                             encode_sequence_length,
                                                             use_decoder_input_ids=use_decoder_input_ids,
                                                             device=device)

            ort_outputs = T5EncoderDecoderInitHelper.onnxruntime_inference(ort_session, inputs)

            # Run inference of PyTorch model
            input_list = inputs.to_list()
            torch_outputs = model(*input_list)

            assert (torch_outputs[0].cpu().numpy().shape == ort_outputs[0].shape)
            max_diff = numpy.amax(numpy.abs(torch_outputs[0].cpu().numpy() - ort_outputs[0]))
            logger.debug(f"logits max_diff={max_diff}")
            max_diff_all = max_diff

            assert (torch_outputs[1].cpu().numpy().shape == ort_outputs[1].shape)
            max_diff = numpy.amax(numpy.abs(torch_outputs[1].cpu().numpy() - ort_outputs[1]))
            logger.debug(f"encoder_hidden_states max_diff={max_diff}")
            max_diff_all = max(max_diff_all, max_diff)

            for i in range(2 * model.config.num_layers):
                max_diff = numpy.amax(numpy.abs(torch_outputs[2][i].cpu().numpy() - ort_outputs[2 + i]))
                logger.debug(f"self attention past state {i} max_diff={max_diff}")

            for i in range(2 * model.config.num_layers):
                max_diff = numpy.amax(
                    numpy.abs(torch_outputs[3][i].cpu().numpy() - ort_outputs[2 + 2 * model.config.num_layers + i]))
                logger.debug(f"cross attention past state {i} max_diff={max_diff}")
                max_diff_all = max(max_diff_all, max_diff)

            test_cases_max_diff.append(max_diff_all)
            logger.info(
                f"batch_size={batch_size} encode_sequence_length={encode_sequence_length}, max_diff={max_diff_all}")

        return max(test_cases_max_diff)
