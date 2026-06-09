from collections.abc import Iterable
from typing import Any, Protocol, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerConfig
from torch.optim import Optimizer


class OptimizerProtocol(Protocol):
    """
    Protocol defining the interface for optimizers used to train the neural vocoder.
    """

    keywords: dict[str, Any]

    def __call__(self, params: Iterable[torch.Tensor] | Iterable[dict[str, Any]]) -> Optimizer: ...


class VocoderLightningModule(LightningModule):
    """
    Lightning module for a neural vocoder.
    """

    def __init__(
        self,
        # encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        discriminator: torch.nn.Module,
        adversarial_criterion: torch.nn.Module,
        feature_map_criterion: torch.nn.Module,
        generator_optimizer: OptimizerProtocol,
        discriminator_optimizer: OptimizerProtocol,
        scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        gradient_accumulation_steps: int = 1,
        reconstruction_criterion: torch.nn.Module | None = None,
        reconstruction_loss_weight: float = 15.0,
    ):
        """
        Initialise the Lightning module.

        :param encoder: SSL encoder used as a feature extractor.
        :type encoder: torch.nn.Module
        :param decoder: Neural vocoder to train
        :type encoder: torch.nn.Module
        :param optimizer: Optimizer class.
        :type optimizer: Type[torch.optim.Optimizer]
        :param scheduler: Optional learning rate scheduler class.
        :type scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]]
        """

        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "adversarial_criterion",
                "feature_map_criterion",
                "discriminator",
                "reconstruction_criterion",
            ],
        )
        self.automatic_optimization = False

        # self.encoder = encoder
        self.decoder = cast(torch.nn.Module, torch.compile(decoder.to(memory_format=torch.channels_last)))  # ty:ignore[no-matching-overload]
        self.discriminator = cast(torch.nn.Module, torch.compile(discriminator.to(memory_format=torch.channels_last)))  # ty:ignore[no-matching-overload]
        self.generator_optimizer_class = generator_optimizer
        self.discriminator_optimizer_class = discriminator_optimizer
        self.scheduler_class = scheduler
        self.adversarial_criterion = torch.compile(adversarial_criterion, mode="reduce-overhead")
        self.feature_map_criterion = torch.compile(feature_map_criterion)

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.reconstruction_criterion = reconstruction_criterion
        self.reconstruction_loss_weight = reconstruction_loss_weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # `inputs` are precomputed encoder features [B, MT, latent_dim]; the decoder expects
        # channels-first [B, latent_dim, MT] and returns (audio, updated_kv_caches).
        latents = inputs.transpose(1, 2)
        # with torch.inference_mode():
        #     inputs = self.encoder(inputs)
        reconstructed_audio, _ = self.decoder(latents)
        return reconstructed_audio

    def training_step(
        self,
        batch,
        batch_idx: int,
    ):
        """
        Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data containing the inputs and targets.
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        """

        inputs, targets = batch

        # Avoid checking whether or not `self.optimizers` is iterable in the hot path, just assume it is
        generator_optimizer, discriminator_optimizer = cast(list[Optimizer], self.optimizers())

        reconstructed_audio = self.forward(inputs)
        logits_for_real_audio, feature_maps_for_real_audio = self.discriminator(targets)

        # Train the generator
        self.toggle_optimizer(generator_optimizer)
        adversarial_loss = torch.tensor(0.0, device=self.device)
        feature_map_loss = torch.tensor(0.0, device=self.device)
        logits_for_generated_audio, feature_maps_for_generated_audio = self.discriminator(reconstructed_audio)
        # Calculate generator adversarial loss
        for sub_discriminator_logits in logits_for_generated_audio:
            adversarial_loss += -sub_discriminator_logits.mean()
        # Calculate generator feature map loss
        num_feature_maps = 0
        for sub_discriminator_feature_maps_for_real_audio, sub_discriminator_feature_maps_for_generated_audio in zip(
            feature_maps_for_real_audio,
            feature_maps_for_generated_audio,
            strict=True,
        ):
            for real_feature_map, generated_feature_map in zip(
                sub_discriminator_feature_maps_for_real_audio,
                sub_discriminator_feature_maps_for_generated_audio,
                strict=True,
            ):
                feature_map_loss += self.feature_map_criterion(real_feature_map.detach(), generated_feature_map)
                num_feature_maps += 1

        # Normalize generator losses
        num_discriminators = len(logits_for_generated_audio)
        adversarial_loss /= num_discriminators
        feature_map_loss /= num_feature_maps
        generator_total_loss = adversarial_loss + feature_map_loss
        reconstruction_loss = torch.tensor(0.0, device=self.device)
        if self.reconstruction_criterion is not None:
            reconstruction_loss = self.reconstruction_criterion(reconstructed_audio, targets)
            generator_total_loss = generator_total_loss + self.reconstruction_loss_weight * reconstruction_loss
        del feature_maps_for_real_audio, logits_for_generated_audio, feature_maps_for_generated_audio
        self.manual_backward(generator_total_loss / self.gradient_accumulation_steps)
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            generator_optimizer.step()
            generator_optimizer.zero_grad(set_to_none=True)
            self.log_dict(
                {
                    "train/generator_adversarial_loss": adversarial_loss.detach(),
                    "train/generator_feature_map_loss": feature_map_loss.detach(),
                    "train/generator_reconstruction_loss": reconstruction_loss.detach(),
                    "train/generator_total_loss": generator_total_loss.detach(),
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.untoggle_optimizer(generator_optimizer)

        # Train the discriminator
        self.toggle_optimizer(discriminator_optimizer)
        logits_for_generated_audio, _feature_maps_for_generated_audio = self.discriminator(reconstructed_audio.detach())
        # How well can the discriminator label real audio as real?
        real_loss = torch.tensor(0.0, device=self.device)
        for sub_discriminator_logits_for_real_audio in logits_for_real_audio:
            real_loss += self.adversarial_criterion(
                sub_discriminator_logits_for_real_audio,
                torch.ones_like(sub_discriminator_logits_for_real_audio),
            )
        # How well can the discriminator label generated audio as generated?
        generated_loss = torch.tensor(0.0, device=self.device)
        for sub_discriminator_logits_for_generated_audio in logits_for_generated_audio:
            generated_loss += self.adversarial_criterion(
                sub_discriminator_logits_for_generated_audio,
                torch.zeros_like(sub_discriminator_logits_for_generated_audio),
            )
        # Normalize discriminator losses and backprop
        num_discriminators = len(logits_for_generated_audio)
        real_loss /= num_discriminators
        generated_loss /= num_discriminators
        discriminator_loss = real_loss + generated_loss
        del _feature_maps_for_generated_audio
        self.manual_backward(discriminator_loss / self.gradient_accumulation_steps)
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad()
            self.log_dict(
                {
                    "train/discriminator_real_loss": real_loss.detach(),
                    "train/discriminator_generated_loss": generated_loss.detach(),
                    "train/discriminator_total_loss": discriminator_loss.detach(),
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.untoggle_optimizer(discriminator_optimizer)

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ):
        """
        Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data containing the inputs and targets.
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        """

        inputs, targets = batch
        reconstructed_audio = self.forward(inputs)
        reconstruction_loss = self.reconstruction_criterion(reconstructed_audio, targets)
        waveform_mae = torch.nn.functional.l1_loss(reconstructed_audio, targets)
        total_loss = reconstruction_loss + waveform_mae

        self.log_dict(
            {
                "valid/reconstruction_loss": reconstruction_loss.detach(),
                "valid/waveform_mae": waveform_mae.detach(),
                "valid/loss": total_loss.detach(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(
        self,
    ) -> list[OptimizerConfig]:
        # Only apply weight decay to transformers' weights
        weight_decay = self.generator_optimizer_class.keywords.get("weight_decay", 0.0)
        parameters_with_weight_decay = []
        parameters_without_weight_decay = []

        for name, param in self.decoder.named_parameters():
            if not param.requires_grad:
                continue
            if "transformer" in name:
                parameters_with_weight_decay.append(param)
            else:
                parameters_without_weight_decay.append(param)

        generator_optimizer = self.generator_optimizer_class(
            params=[
                {"params": parameters_with_weight_decay, "weight_decay": weight_decay},
                {"params": parameters_without_weight_decay, "weight_decay": 0.0},
            ],
        )
        discriminator_optimizer = self.discriminator_optimizer_class(params=self.discriminator.parameters())

        gen_config = cast(OptimizerConfig, {"optimizer": generator_optimizer})
        disc_config = cast(OptimizerConfig, {"optimizer": discriminator_optimizer})

        return [gen_config, disc_config]
