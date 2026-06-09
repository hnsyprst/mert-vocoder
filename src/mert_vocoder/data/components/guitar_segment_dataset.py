import ast
import csv
import logging
import math
import sys
from dataclasses import dataclass
from os import R_OK, access
from pathlib import Path

import numpy as np
import torch
from soundfile import SoundFile
from torchaudio.transforms import Resample

log = logging.getLogger(__name__)

# The metadata CSV stores a very large `frame_activity` field per row.
# Raise the field size limit so the `csv` module can read it.
csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class Segment:
    """A single contiguous segment of instrument activity within one stem."""

    path: Path
    # Read range in the source file's native sample domain (before any resampling).
    start_sample: int
    end_sample: int

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample


class GuitarSegmentDataset(torch.utils.data.Dataset):
    """
    Dataset of guitar stems read from a splitter metadata CSV.

    The metadata CSV describes one audio stem per row, including a `frame_activity` column:
    a list of booleans flagging, for each ~1 second frame, whether the instrument is active.
    This dataset filters the rows down to a single `instrument_type` (``"Guitar"`` by default)
    and treats **each contiguous run of active frames as a single sample**. A stem with several
    separated bursts of playing therefore contributes several samples; a stem that is silent
    throughout contributes none.

    Each item is read on the fly from disk, optionally resampled and channel-normalised.
    If `audio_duration` is set, segments longer than the duration are randomly cropped to it and
    shorter segments are right-padded with near-zero noise, so items can be batched together.
    If `audio_duration` is `None`, the full (variable-length) segment is returned.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        sample_rate: int,
        num_channels: int,
        audio_root: str | Path | None = None,
        file_extension: str = "flac",
        audio_duration: float | None = None,
        instrument_type: str = "Guitar",
        min_active_frames: int = 1,
        strict_validation: bool = True,
    ):
        """
        Initialise the dataset.

        Audio is located by directory structure, not by the metadata's `filename` column (which
        points at the original, unsplit sources). For each row the stem is read from
        `<audio_root>/<split>/<song>/<name>.<file_extension>`.

        :param metadata_csv: Path to the splitter metadata CSV (e.g. `train_metadata.csv`).
        :type metadata_csv: Union[str, Path]
        :param sample_rate: Sample rate of the returned audio. Files not matching this sample rate
            are resampled on the fly.
        :type sample_rate: int
        :param audio_root: Root directory containing the per-split stem directories. Defaults to the
            directory containing `metadata_csv` (the split folders sit alongside the metadata CSVs).
        :type audio_root: Optional[Union[str, Path]]
        :param file_extension: Extension of the on-disk stem files (without the dot). The labelled
            split stores every stem as `flac` regardless of the metadata `ext` column, so this
            defaults to `"flac"`.
        :type file_extension: str
        :param num_channels: Number of channels in each returned item.
            Items with fewer channels have channels added by repeating,
            items with more channels have channels removed by averaging.
        :type num_channels: int
        :param audio_duration: Duration (in seconds) to crop/pad each segment to. If `None`, the
            full variable-length segment is returned (only batchable with `batch_size=1` or a
            custom collate function).
        :type audio_duration: Optional[float]
        :param instrument_type: Value of the `instrument_type` column to keep. Defaults to `"Guitar"`.
        :type instrument_type: str
        :param min_active_frames: Minimum length (in frames, i.e. ~seconds) of an activity run for it
            to be included as a sample. Use to drop very short bursts.
        :type min_active_frames: int
        :param strict_validation: If True, raises if any referenced file is missing or unreadable.
            Otherwise, those files are skipped.
        :type strict_validation: bool
        """

        super().__init__()

        metadata_csv = Path(metadata_csv)
        if not metadata_csv.is_file():
            raise ValueError(f"`metadata_csv` not found. Got {metadata_csv=}")
        audio_root = Path(audio_root) if audio_root is not None else metadata_csv.parent
        if not audio_root.is_dir():
            raise ValueError(f"`audio_root` is not a directory. Got {audio_root=}")
        if audio_duration is not None and audio_duration <= 0:
            raise ValueError(f"`audio_duration` must be positive or None. Got {audio_duration=}")
        if min_active_frames < 1:
            raise ValueError(f"`min_active_frames` must be >= 1. Got {min_active_frames=}")

        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.audio_duration = audio_duration
        self.instrument_type = instrument_type
        self.min_active_frames = min_active_frames

        # torchaudio Resampler objects can reuse the same kernel when resampling with the same
        # parameters, so we cache one resampler per source sample rate for a significant speedup.
        self.resampler_cache: dict[int, Resample] = dict()

        log.info(f"Setting up dataset from {metadata_csv} (instrument_type={instrument_type!r}).")

        self.segments: list[Segment] = []
        files_errors: dict[Path, str] = dict()
        seen_paths: set[Path] = set()
        with open(metadata_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["instrument_type"] != instrument_type:
                    continue

                # Locate the stem by directory structure rather than the `filename` column.
                path = audio_root / row["split"] / row["song"] / f"{row['name']}.{file_extension}"
                # Validate each referenced file once.
                if path not in seen_paths:
                    seen_paths.add(path)
                    if not path.is_file():
                        files_errors[path] = "Not a file."
                    elif not access(path, R_OK):
                        files_errors[path] = "Did not have permission to read file."
                if path in files_errors:
                    continue

                length_samples = int(float(row["length_samples"]))
                frame_activity = np.asarray(ast.literal_eval(row["frame_activity"]), dtype=bool)
                if frame_activity.size == 0:
                    continue

                # Frames are uniformly spaced across the file, so map frame indices to samples
                # using this stem's own (samples per frame) ratio rather than assuming a frame rate.
                samples_per_frame = length_samples / frame_activity.size
                for start_frame, end_frame in self._contiguous_runs(frame_activity):
                    if end_frame - start_frame < self.min_active_frames:
                        continue
                    start_sample = round(start_frame * samples_per_frame)
                    end_sample = min(round(end_frame * samples_per_frame), length_samples)
                    if end_sample > start_sample:
                        self.segments.append(Segment(path, start_sample, end_sample))

        if files_errors:
            for path, error in files_errors.items():
                log.warning(f"Skipping unreadable file: {path} ({error})")
            if strict_validation:
                raise RuntimeError(
                    f"Found {len(files_errors)} files in the metadata failing read validation. "
                    "Since `strict_validation` is True, this is an error. See above logs."
                )

        if not self.segments:
            raise ValueError(f"Found no activity segments for instrument_type={instrument_type!r} in {metadata_csv}.")
        log.info(
            f"Finished initialising dataset. Found {len(self.segments)} activity segments "
            f"across {len(seen_paths) - len(files_errors)} stems."
        )

    @staticmethod
    def _contiguous_runs(frame_activity: np.ndarray) -> list[tuple[int, int]]:
        """Return `(start_frame, end_frame)` half-open spans of each contiguous run of `True`."""
        # Pad with False on both sides so every run has a rising and falling edge.
        padded = np.concatenate(([False], frame_activity, [False]))
        edges = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        return list(zip(starts.tolist(), ends.tolist(), strict=True))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> torch.Tensor:
        segment = self.segments[index]
        audio, sample_rate = self._read_segment(segment)

        if sample_rate != self.sample_rate:
            if not self.resampler_cache.get(sample_rate):
                self.resampler_cache[sample_rate] = Resample(sample_rate, self.sample_rate, dtype=audio.dtype)
            audio = self.resampler_cache[sample_rate](audio)

        if audio.shape[0] != self.num_channels:
            if audio.shape[0] < self.num_channels:
                # Upmix by repeating channels cyclically (e.g. mono -> stereo)
                repeats = math.ceil(self.num_channels / audio.shape[0])
                audio = audio.repeat((repeats, 1))[: self.num_channels]
            else:
                # Downmix by averaging interleaved channel groups
                # e.g. 6ch -> 2ch: avg([0,2,4]), avg([1,3,5])
                audio = torch.stack([audio[i :: self.num_channels].mean(0) for i in range(self.num_channels)])
        return audio

    def _read_segment(self, segment: Segment) -> tuple[torch.Tensor, int]:
        """
        Read a segment from disk in the source file's native sample domain.

        When `audio_duration` is set, the read is cropped to that many native samples from a random
        offset within the segment, or padded with near-zero noise if the segment is shorter.
        """
        file = SoundFile(segment.path)
        seg_len = segment.num_samples
        want = int(self.audio_duration * file.samplerate) if self.audio_duration is not None else seg_len

        if seg_len > want:
            # Randomly crop to `want` samples from somewhere inside the segment.
            offset = segment.start_sample + int(np.random.uniform() * (seg_len - want))
            read_len = want
        else:
            offset = segment.start_sample
            read_len = seg_len

        # Initialise with near-zero noise so any padding (when `want > read_len`) is not pure silence.
        out = np.random.normal(size=(want, file.channels)).astype(np.float32) * 1e-8
        file.seek(offset)
        file.read(read_len, out=out[:read_len])
        return torch.from_numpy(out).T, file.samplerate
