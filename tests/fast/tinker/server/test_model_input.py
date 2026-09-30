"""Image chunks retain token order, tensor shapes, and SDK label alignment."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw
from miles.tinker.runtime import MilesBackend, _build_train_data, _pad_to_dp_multiple
from miles.tinker.server.encoding import decode_command, decode_sample_request
from miles.tinker.server.proto_codec import decode_forward_backward_request
from tinker import types
from tinker.proto.request_conv import forward_backward_request_to_proto


class ImageProcessor:
    image_token = "<image>"

    def __call__(self, *, text, images, add_special_tokens, return_tensors):
        assert text == self.image_token and not add_special_tokens and return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[99, 99]]),
            "attention_mask": torch.ones(1, 2),
            "mm_token_type_ids": torch.ones(1, 2),
            "pixel_values": torch.tensor([list(images[0].getpixel((0, 0)))], dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
        }


def _image(color, expected_tokens=2):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return types.ImageChunk(data=buffer.getvalue(), format="png", expected_tokens=expected_tokens)


def _request(*, expected_tokens=2):
    return types.ForwardBackwardRequest(
        model_id="image-model",
        seq_id=7,
        forward_backward_input=types.ForwardBackwardInput(
            loss_fn="cross_entropy",
            data=[
                types.Datum(
                    model_input=types.ModelInput(
                        chunks=[
                            types.EncodedTextChunk(tokens=[1]),
                            _image("red", expected_tokens),
                            types.EncodedTextChunk(tokens=[2]),
                            _image("blue"),
                            types.EncodedTextChunk(tokens=[3]),
                        ]
                    ),
                    loss_fn_inputs={"target_tokens": [0, 0, 0, 0, 0, 0, 4], "weights": [0, 0, 0, 0, 0, 0, 1]},
                )
            ],
        ),
    )


def _decode(request, transport, processor, forward_only=False):
    if transport == "protobuf":
        message = forward_backward_request_to_proto(request)
        message.forward_only = forward_only
        return decode_forward_backward_request(message.SerializeToString(), processor)
    payload = {
        "model_id": request.model_id,
        "seq_id": request.seq_id,
        "forward_only": forward_only,
        "forward_backward_input": {
            "loss_fn": request.forward_backward_input.loss_fn,
            "data": [
                {
                    "model_input": datum.model_input.model_dump(mode="json"),
                    "loss_fn_inputs": {key: value.data for key, value in datum.loss_fn_inputs.items()},
                }
                for datum in request.forward_backward_input.data
            ],
        },
    }
    return decode_command("forward_backward", payload, processor)


@pytest.mark.parametrize("transport", ["json", "protobuf"])
@pytest.mark.parametrize("forward_only", [False, True])
def test_sdk_images_preserve_order_and_explicit_targets(transport, forward_only):
    op, decoded = _decode(_request(), transport, ImageProcessor(), forward_only)
    assert op == ("forward_only" if forward_only else "forward_backward")
    datum = decoded["datums"][0]
    assert datum["tokens"] == [1, 99, 99, 2, 99, 99, 3, 4]
    assert datum["target_tokens"] == [0, 0, 0, 0, 0, 0, 4]
    assert datum["weights"] == [0, 0, 0, 0, 0, 0, 1]
    tensors = datum["multimodal_train_inputs"]
    assert set(tensors) == {"pixel_values", "image_grid_thw"}
    assert tensors["pixel_values"].tolist() == [[255, 0, 0], [0, 0, 255]]
    assert tensors["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 4]]


@pytest.mark.parametrize("transport", ["json", "protobuf"])
@pytest.mark.parametrize("problem", ["token_count", "text_model", "invalid_image", "target_length"])
def test_invalid_images_produce_ordered_user_errors(transport, problem):
    request = _request(expected_tokens=3 if problem == "token_count" else 2)
    if problem == "invalid_image":
        request.forward_backward_input.data[0].model_input.chunks[1] = types.ImageChunk(
            data=b"not an image", format="png"
        )
    if problem == "target_length":
        request.forward_backward_input.data[0].loss_fn_inputs["target_tokens"] = types.TensorData.from_numpy(
            np.array([4])
        )
    _, decoded = _decode(request, transport, None if problem == "text_model" else ImageProcessor())
    assert decoded["model_id"] == "image-model" and decoded["seq_id"] == 7
    assert decoded["validation_error"]


def test_sampling_passes_images_to_sglang():
    payload = types.SampleRequest(
        prompt=_request().forward_backward_input.data[0].model_input,
        sampling_params=types.SamplingParams(max_tokens=2),
    ).model_dump(mode="json")
    decoded = decode_sample_request(payload, ImageProcessor())
    request = MilesBackend(None, "http://router")._generate_request(decoded, lora_name=None)
    assert request["input_ids"] == [1, 99, 99, 2, 99, 99, 3]
    assert len(request["image_data"]) == 2
    assert request["image_data"][0] != request["image_data"][1]
    assert all(value.startswith("data:image/png;base64,") for value in request["image_data"])


@pytest.mark.parametrize("field", ["pixel_values", "input_features", "mm_audio_dmel"])
def test_image_and_audio_tensors_survive_mixed_batch_padding_and_dp_split(field):
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    text = {"tokens": [1, 2], "target_tokens": [2], "target_len": 1}
    multimodal = dict(text, multimodal_train_inputs={field: tensor})
    batch = _build_train_data(_pad_to_dp_multiple([(0, text), (1, multimodal), (1, multimodal)], 2))
    assert batch["multimodal_train_inputs"][0] is None
    shards = split_train_data_by_dp_raw(SimpleNamespace(balance_data=False, multi_lora_n_adapters=2), batch, dp_size=2)
    for shard in shards:
        for index, tensors in zip(shard["sample_indices"], shard["multimodal_train_inputs"], strict=True):
            if index == 0:
                assert tensors is None
            else:
                torch.testing.assert_close(tensors[field], tensor)
    assert batch["loss_masks"][-1] == [0]
    assert "multimodal_train_inputs" not in text
