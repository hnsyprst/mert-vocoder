import logging
from os import R_OK, access
from pathlib import Path

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.table import Table

from mert_vocoder.utils.audio import audioread

log = logging.getLogger(__name__)


class PrecomputedFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset of precomputed SSL features paired with their target audio.

    Reads the output of `scripts/compute_mert_embeddings.py`: a flat directory holding, for each
    item, a `<stem>.<feature_extension>` tensor of encoder features and a matching
    `<stem>.<audio_extension>` audio file. Each item is returned as a `(features, audio)` tuple,
    where `features` is the precomputed encoder output (the decoder input, shape `(frames, hidden)`)
    and `audio` is the reconstruction target (shape `(num_channels, samples)`). This lets the vocoder
    be trained on cached features without re-running the (frozen) SSL encoder every step.
    """

    def __init__(
        self,
        data_dir: str | Path,
        feature_extension: str = "pt",
        audio_extension: str = "wav",
        strict_validation: bool = True,
    ):
        """
        Initialise the dataset.

        :param data_dir: Directory of `<stem>.<feature_extension>` / `<stem>.<audio_extension>` pairs,
            as written by `scripts/compute_mert_embeddings.py`.
        :type data_dir: Union[str, Path]
        :param feature_extension: Extension of the saved feature tensors (without the dot).
        :type feature_extension: str
        :param audio_extension: Extension of the saved audio files (without the dot).
        :type audio_extension: str
        :param strict_validation: If True, raises if any feature file has no readable matching audio
            file. Otherwise, those pairs are excluded from the dataset.
            Disable this with care; if you fix failing files later, you will break reproducibility for your run.
        :type strict_validation: bool
        """

        super().__init__()

        log.info("Setting up dataset.")

        data_dir = Path(data_dir)
        if not data_dir.exists(follow_symlinks=True):
            raise ValueError(f"`data_dir` not found. Got {data_dir=}")
        if not data_dir.is_dir():
            raise ValueError(f"`data_dir` must be a directory. Got {data_dir=}")

        self.feature_extension = feature_extension.lstrip(".")
        self.audio_extension = audio_extension.lstrip(".")

        # Suppress fancy console status output if not rank zero to avoid glitchy output
        console = Console(quiet=rank_zero_only.rank != 0)  # ty:ignore[unresolved-attribute]

        # Discover feature tensors; each is paired with its audio file by shared stem.
        with console.status(f"Reading dataset from {data_dir}...", spinner="earth"):
            feature_paths = sorted(data_dir.glob(f"*.{self.feature_extension}"))
        if not feature_paths:
            raise ValueError(f"Found no `*.{self.feature_extension}` feature files in `data_dir` ({data_dir=}).")

        # Check each feature file has a readable matching audio file.
        with console.status("Checking all files for read issues...", spinner="earth"):
            pairs: list[tuple[Path, Path]] = []
            files_errors: dict[Path, str] = dict()
            for feature_path in feature_paths:
                audio_path = feature_path.with_suffix(f".{self.audio_extension}")
                if not access(feature_path, R_OK):
                    files_errors[feature_path] = "Did not have permission to read feature file."
                elif not audio_path.is_file():
                    files_errors[feature_path] = f"No matching audio file ({audio_path.name})."
                elif not access(audio_path, R_OK):
                    files_errors[feature_path] = f"Did not have permission to read matching audio file ({audio_path.name})."
                else:
                    pairs.append((feature_path, audio_path))
        if files_errors:
            table = Table(
                "Feature file",
                "Error",
                row_styles=["dim", ""],
                title=f"{len(files_errors)} feature files in the dataset failed read validation:",
                title_justify="left",
                title_style="bold red",
            )
            for path, error in files_errors.items():
                table.add_row(str(path), error)
            log.warning(table)
            if strict_validation:
                raise RuntimeError(
                    f"Found {len(files_errors)} feature files in the dataset failing read validation. "
                    "Since `strict_validation` is True, this is an error. "
                    "See above logs for list of erroneous files."
                )
            else:
                log.info(
                    f"Found {len(files_errors)} feature files in the dataset failing read validation. "
                    "Since `strict_validation` is False, will skip these files."
                )

        self.pairs: list[tuple[Path, Path]] = pairs
        log.info(f"Finished initialising dataset. Dataset contains {len(self.pairs)} feature/audio pairs.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature_path, audio_path = self.pairs[index]
        features = torch.load(feature_path)
        audio, _ = audioread(audio_path)
        return features, audio
