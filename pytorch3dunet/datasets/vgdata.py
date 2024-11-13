from dataclasses import dataclass
import glob
import gzip
from itertools import chain

import numpy as np
from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.utils import (
    ConfigDataset,
    calculate_stats,
    get_slice_builder,
    mirror_pad,
)
from pytorch3dunet.unet3d.utils import get_logger
from typing import List
import json
import os


logger = get_logger("VGDataset")


@dataclass
class Label:
    dtype: str
    file_path: str
    is_compressed: bool
    name: str
    order: str
    sampling_distance_mm: List[float]
    shape: List[int]
    threshold: int


@dataclass
class RawData:
    dtype: str
    file_path: str
    order: str
    original_data_range: List[float]
    sampling_distance_mm: List[float]
    shape: List[int]


@dataclass
class LabeledVolumeDescription:
    export_time: str
    file_format_version: int
    labels: List[Label]
    raw_data: RawData
    vgl_project: str

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    @staticmethod
    def from_json(data: str):
        data_dict = json.loads(data)
        data_dict["labels"] = [Label(**label) for label in data_dict["labels"]]
        data_dict["raw_data"] = RawData(**data_dict["raw_data"])
        return LabeledVolumeDescription(**data_dict)


def _create_padded_indexes(indexes, halo_shape):
    return tuple(
        slice(index.start, index.stop + 2 * halo)
        for index, halo in zip(indexes, halo_shape)
    )


def traverse_labeled_volume_paths(file_paths):
    assert isinstance(file_paths, list)
    results = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # if file path is a directory, look in all subdirectories for labeled_volume.json files
            iters = [
                glob.glob(
                    os.path.join(file_path, "**", "labeled_volume.json"), recursive=True
                )
            ]
            for fp in chain(*iters):
                results.append(fp)
        else:
            results.append(file_path)
    return results


class VGDataset(ConfigDataset):
    """
    Dataset for the VG dataset. The dataset is generated from the labeled_volume.json file,
    the output format of VGSTUDIO MAX.

    Args:
        file_path (str): The path to the labeled_volume.json file.
        phase (str): The phase of the dataset. One of "train", "val", or "test".
        slice_builder_config (dict): The configuration for the slice builder.
        transformer_config (dict): The configuration for the transformer.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config):
        assert os.path.isfile(file_path), f"{file_path} is not a file"
        assert phase in ["train", "val", "test"]

        self.phase = phase
        self.file_path = os.path.dirname(file_path)

        self.halo_shape = slice_builder_config.get("halo_shape", [0, 0, 0])

        stats = calculate_stats(None, True)
        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != "test":
            self.label_transform = self.transformer.label_transform()
        else:
            # compare patch and stride configuration
            patch_shape = slice_builder_config.get("patch_shape")
            stride_shape = slice_builder_config.get("stride_shape")
            if sum(self.halo_shape) != 0 and patch_shape != stride_shape:
                logger.warning(
                    f"Found non-zero halo shape {self.halo_shape}. "
                    f"In this case: patch shape and stride shape should be equal for optimal prediction "
                    f"performance, but found patch_shape: {patch_shape} and stride_shape: {stride_shape}!"
                )

        raw, label = self._load_data(file_path)
        assert raw.ndim in [3, 4], "Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)"
        assert label.ndim in [3, 4], "Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)"

        self.raw = raw
        self.label = None if phase == "test" else label
        self.raw_padded = None

        slice_builder = get_slice_builder(raw, label, None, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f"Number of patches: {self.patch_count}")

    def __len__(self):
        return self.patch_count

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]
        if self.phase == "test":
            if len(raw_idx) == 4:
                # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
                raw_idx = raw_idx[
                    1:
                ]  # Remove the first element if raw_idx has 4 elements
                raw_idx_padded = (slice(None),) + _create_padded_indexes(
                    raw_idx, self.halo_shape
                )
            else:
                raw_idx_padded = _create_padded_indexes(raw_idx, self.halo_shape)

            raw_patch_transformed = self.raw_transform(
                self.get_raw_padded_patch(raw_idx_padded)
            )
            return raw_patch_transformed, raw_idx
        else:
            raw_patch_transformed = self.raw_transform(self.get_raw_patch(raw_idx))

            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(
                self.get_label_patch(label_idx)
            )

            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load slice builder config
        slice_builder_config = phase_config["slice_builder"]
        # load files to process
        file_paths = phase_config["file_paths"]
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = traverse_labeled_volume_paths(file_paths)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f"Loading {phase} set from: {file_path}...")
                dataset = cls(
                    file_path=file_path,
                    phase=phase,
                    slice_builder_config=slice_builder_config,
                    transformer_config=transformer_config,
                )
                datasets.append(dataset)
            except Exception:
                logger.error(f"Skipping {phase} set: {file_path}", exc_info=True)
        return datasets

    def volume_shape(self):
        return self.raw.shape

    def get_raw_patch(self, idx):
        return self.raw[idx]

    def get_label_patch(self, idx):
        return self.label[idx]

    def get_raw_padded_patch(self, idx):
        if self.raw_padded is None:
            self.raw_padded = mirror_pad(self.raw, self.halo_shape)
        return self.raw_padded[idx]

    def _open_file(self, filepath, is_compressed):
        """Opens a file based on whether it is gzipped or not."""
        if is_compressed:
            return gzip.open(filepath, "rb")  # Open in binary mode for gzip
        else:
            return open(filepath, "rb")  # Standard open for uncompressed files

    def _load_volume(self, file_path, dtype, shape, order, is_compressed=False):
        """
        Load a volume from a file.
        Args:
            file_path (str): The path to the file containing the volume data.
            dtype (str): The data type of the volume.
            shape (tuple): The shape of the volume.
            order (str): The order of the volume data.
            is_compressed (bool, optional): Whether the volume data is compressed. Defaults to False.
        Returns:
            numpy.ndarray: The loaded volume.
        """
        with self._open_file(file_path, is_compressed) as volume_file:
            volume = np.frombuffer(volume_file.read(), dtype=dtype)
            volume = volume.reshape(shape, order=order)
            # reorder the axis from X,Y,Z to Z,Y,X
            volume = np.moveaxis(volume, [0, 1, 2], [2, 1, 0])

        return volume

    def _load_data(self, file_path):
        volume_dir = os.path.dirname(file_path)
        dataset_description = LabeledVolumeDescription.from_json(open(file_path).read())
        raw_data = dataset_description.raw_data
        raw_path = os.path.join(volume_dir, raw_data.file_path)

        raw_volume = self._load_volume(
            raw_path,
            raw_data.dtype,
            raw_data.shape,
            raw_data.order,
            is_compressed=False,
        )
        label_volumes = []
        for label in dataset_description.labels:
            label_path = os.path.join(volume_dir, label.file_path)
            label_volume = self._load_volume(
                label_path, label.dtype, label.shape, label.order, label.is_compressed
            )
            label_volume = np.where(label_volume > 127, 1, 0).astype(label.dtype)
            label_volumes.append(label_volume)

        label_volume = np.stack(label_volumes, axis=0)

        return raw_volume, label_volume
