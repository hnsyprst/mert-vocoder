import logging
import math
import os
from pathlib import Path

import hydra
import rootutils
import soundfile as sf
import torch
from omegaconf import DictConfig
from rich.progress import track
from torch.utils.data import DataLoader

from mert_vocoder.utils.resolvers import register_resolvers

# Log full error tracebacks
os.environ["HYDRA_FULL_ERROR"] = "1"
# Set PYTHONPATH based on `.project-root` file location
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# Set up logger, capture warnings
logging.captureWarnings(True)
log = logging.getLogger(__name__)


register_resolvers()


@hydra.main(version_base="1.3", config_path="../configs", config_name="compute_mert_embeddings.yaml")
def main(cfg: DictConfig):
    log.info("Computing MERT embeddings.")

    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(cfg.device).eval()

    # MERT's valid (unpadded) convolutions can't emit a frame for the trailing audio samples that
    # don't fill a complete kernel window, so the decoder reconstructs only `num_frames *
    # samples_per_frame` samples -- fewer than were input. We trim each saved target to that length
    # below so the cached audio stays aligned with its features (and the decoder's output length).
    # `samples_per_frame` is the encoder's total downsampling factor (product of its conv strides).
    samples_per_frame = math.prod(model.model.config.conv_stride)

    # The dataset yields audio tensors already resampled to the model's sample rate.
    # Fixed-length items (`audio_duration` set) let us batch the forward pass for speed.
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    log.info(f"Computing features for {len(dataset)} item(s) in batches of {cfg.batch_size}.")
    log.info(f"Writing features and audio to {output_root}.")
    index = 0
    for audio_batch in track(dataloader):
        # (batch, num_channels, frames) -> mono (batch, frames) for the MERT preprocessor
        audio_batch = audio_batch.mean(dim=1)

        with torch.inference_mode():
            features_batch = model(audio_batch.numpy(force=True))  # (batch, frames, hidden)

        # Write each item's features and audio side by side with matching stems
        # so they can be paired back up when read as a dataset later
        for audio, features in zip(audio_batch.cpu(), features_batch.cpu(), strict=True):
            num_samples = features.shape[0] * samples_per_frame
            torch.save(features.clone(), output_root / f"{index:06d}.pt")
            sf.write(output_root / f"{index:06d}.wav", audio[:num_samples].numpy(force=True), model.sample_rate)
            index += 1


if __name__ == "__main__":
    # Catch all errors to enable error logging via logger
    try:
        main()
    except Exception as e:
        log.exception(e)
