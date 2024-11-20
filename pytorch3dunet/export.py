from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List
import torch
import torch.onnx

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model

logger = utils.get_logger("UNet3DOnnxExport")


@dataclass
class PeakNormalization:
    type: str
    air_peak_target_value: float
    material_peak_target_value: float

@dataclass
class ZeroMeanUnitVariancePerSample:
    type: str

@dataclass
class InferenceBlockSize:
    min_block_size: List[int]
    opt_block_size: List[int]
    max_block_size: List[int]


@dataclass
class BlockwiseStrategy:
    type: str
    block_overlap: List[int]


@dataclass
class Outputs:
    type: str
    class_name_per_channel: List[str]
    region_of_interest_outputs: List[str]


@dataclass
class ModelConfig:
    file_format_version: int
    model_name: str
    model_version: str
    model_description: str
    preprocessing: List[PeakNormalization | ZeroMeanUnitVariancePerSample]
    inference_block_size: InferenceBlockSize
    blockwise_strategy: BlockwiseStrategy
    outputs: Outputs

def create_model_config(model_path, version, patch_shape, halo_shape, out_channels):
    model_name = Path(model_path).stem
    output_classes = ["Class " + str(i) for i in range(1, out_channels + 1)]
    model_config = ModelConfig(
        file_format_version=1,
        model_name=model_name,
        model_version=version,
        model_description=model_name,
        preprocessing=[
            ZeroMeanUnitVariancePerSample(
                type="zero_mean_unit_variance_per_sample"
            )
        ],
        inference_block_size=InferenceBlockSize(
            min_block_size=patch_shape,
            opt_block_size=patch_shape,
            max_block_size=patch_shape
        ),
        blockwise_strategy=BlockwiseStrategy(
            type="blockwise_with_context",
            block_overlap=halo_shape
        ),
        outputs=Outputs(
            type="class_probabilities",
            class_name_per_channel=output_classes,
            region_of_interest_outputs=output_classes
        )
    )
    # Save the model configuration to a JSON file with the same name as the model at the same location
    model_config_path = Path(model_path).with_suffix(".info.json")
    with open(model_config_path, "w") as f:
        json.dump(asdict(model_config), f, indent=4)

def main():
    # Load configuration
    config, _ = load_config()

    # Create the model
    model = get_model(config["model"])

    # Load model state
    model_path = config["model_path"]
    logger.info(f"Loading model from {model_path}...")
    utils.load_checkpoint(model_path, model)

    # Get test data loader
    loader_config = config["loaders"]
    test_loader_config = loader_config["test"]
    slice_builder_config = test_loader_config.get("slice_builder", None)
    patch_shape = slice_builder_config.get("patch_shape")
    halo_shape = slice_builder_config.get("halo_shape")

    onnx_output_path = loader_config.get("output_dir", None) + "/model.onnx"
    out_channels = config['model'].get('out_channels')

    dummy_input = torch.randn(1, 1, *patch_shape)

    torch.onnx.export(
        model,  # The wrapped model
        dummy_input,  # Dummy input for the ONNX export
        onnx_output_path,  # Path to save the ONNX model
        export_params=True,  # Store the trained parameter weights
        opset_version=17,  # ONNX version
        do_constant_folding=False,  # Optimize model by folding constants
        input_names=["input"],  # Input tensor name
        output_names=["output"],  # Output tensor name
    )

    create_model_config(onnx_output_path, "1.0", patch_shape, halo_shape, out_channels)


if __name__ == "__main__":
    main()
