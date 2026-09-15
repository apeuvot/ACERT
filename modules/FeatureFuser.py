import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from transformers import Wav2Vec2ForPreTraining, HubertModel, WavLMModel


def prepare_mask(length, shape, dtype, device):
    # Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype, device=device
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[
        (torch.arange(mask.shape[0], device=device), length - 1)
    ] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask


class Wav2vec2Wrapper(nn.Module):
    """Self-supervised speech encoder (wav2vec 2.0 / HuBERT / WavLM).

    The convolutional feature extractor stays frozen (wrapped in ``torch.no_grad``);
    only the transformer encoder is fine-tuned, see :meth:`trainable_params`.
    SpecAugment (time and feature masking) is applied during training only.

    The paper uses ``facebook/hubert-large-ll60k``.
    """

    def __init__(self, model_path):
        super().__init__()

        if "wav2vec2" in model_path.lower():
            self.wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(model_path).wav2vec2
        elif "hubert" in model_path.lower():
            self.wav2vec2 = HubertModel.from_pretrained(model_path)
        elif "wavlm" in model_path.lower():
            self.wav2vec2 = WavLMModel.from_pretrained(model_path)
        else:
            raise ValueError(f"Model not supported: {model_path}")

        # Disable gradient checkpointing for ddp
        if hasattr(self.wav2vec2, "encoder") and hasattr(self.wav2vec2.encoder, "config"):
            self.wav2vec2.encoder.config.gradient_checkpointing = False

        # SpecAug
        self.mask_time_length = 15
        self.mask_time_prob = 0.08
        self.observe_time_prob = 0.0

        self.mask_feature_length = 64
        self.mask_feature_prob = 0.05

    def trainable_params(self):
        return list(self.wav2vec2.encoder.parameters())

    def forward(self, x, length=None):
        # INPUT: (B, T) raw waveform
        # OUTPUT: (B, T', C) frame-level representations
        with torch.no_grad():
            x = self.wav2vec2.feature_extractor(x)
            x = x.transpose(1, 2)
            # feature_projection returns a tuple for wav2vec2 and a tensor for hubert
            feats = self.wav2vec2.feature_projection(x)
            if isinstance(feats, tuple):
                x, _ = feats
            else:
                x = feats

            mask = None
            if length is not None:
                length = self.get_feat_extract_output_lengths(length)
                mask = prepare_mask(length, x.shape[:2], x.dtype, x.device)

            if self.training:
                batch_size, sequence_length, hidden_size = x.size()

                # apply SpecAugment along time axis
                if self.mask_time_prob > 0:
                    mask_time_indices = torch.tensor(
                        _compute_mask_indices(
                            (batch_size, sequence_length),
                            self.mask_time_prob,
                            self.mask_time_length,
                            min_masks=2,
                        ),
                        dtype=torch.bool,
                        device=x.device
                    )

                    masked_indicies = mask_time_indices & mask
                    flip_mask = torch.rand((batch_size, sequence_length), device=masked_indicies.device) > self.observe_time_prob
                    x[masked_indicies & flip_mask] = self.wav2vec2.masked_spec_embed.to(x.dtype)

                # apply SpecAugment along feature axis
                if self.mask_feature_prob > 0:
                    mask_feature_indices = torch.tensor(
                        _compute_mask_indices(
                            (batch_size, hidden_size),
                            self.mask_feature_prob,
                            self.mask_feature_length,
                            min_masks=1
                        ),
                        dtype=torch.bool,
                        device=x.device
                    )
                    x[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0

        x = self.wav2vec2.encoder(x, attention_mask=mask)[0]
        return F.relu(x)

    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.wav2vec2.config.conv_kernel, self.wav2vec2.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length
