import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.FeatureFuser import Wav2vec2Wrapper


class BaselineHead(nn.Module):
    """Utterance-level SER model, without conversational context.

    The SSL encoder, a temporal average pooling over the non-padded frames and a
    classification head.
    """

    def __init__(self, n_outputs, model_path):
        super().__init__()
        self.ssl = Wav2vec2Wrapper(model_path)
        feature_dim = self.ssl.wav2vec2.config.hidden_size

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feature_dim, n_outputs)
        )

    def trainable_params(self):
        return list(self.classifier.parameters()) + self.ssl.trainable_params()

    def forward(self, x, length):
        reps = self.ssl(x, length)
        last_feat_pos = self.ssl.get_feat_extract_output_lengths(length) - 1
        logits = reps.permute(1, 0, 2)  # L, B, C
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0), -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        logits = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        return self.classifier(logits)


class ACERT(nn.Module):
    """Averaged Contextual Emotion Representation through Time.

    The input waveform holds the target utterance surrounded by its conversational
    context (built by :class:`~downstream.Custom.dataloader.CustomEmoContextDataset`);
    ``target_start`` / ``target_end`` mark the target inside that waveform.

    Frame-level features H come from the SSL encoder followed by a linear layer and
    a ReLU. H is split into the target frames H_t and the context frames H_c; H_c is
    reduced to a single vector c by temporal mean pooling, c is added to every target
    frame, and a feed-forward network follows, both steps with a residual connection
    and a LayerNorm. Temporal average pooling over the enriched target frames gives
    the utterance representation fed to the classification head.
    """

    def __init__(self, n_outputs, model_path, fusion_dim=256):
        super().__init__()
        self.ssl = Wav2vec2Wrapper(model_path)
        hidden_dim = self.ssl.wav2vec2.config.hidden_size

        self.proj = nn.Linear(hidden_dim, fusion_dim)

        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fusion_dim, n_outputs)
        )

    def trainable_params(self):
        return list(self.parameters())

    def forward(self, audio, lengths, target_start, target_end):
        """
        Args:
            audio: (B, T_audio) - raw waveform, context + target concatenated
            lengths: (B,) - audio lengths in samples
            target_start: (B,) - first sample of the target utterance
            target_end: (B,) - last sample of the target utterance

        Returns:
            logits: (B, n_outputs)
        """
        reps = self.ssl(audio, lengths)
        reps = F.relu(self.proj(reps))  # (B, T', D)

        B, T, D = reps.shape

        # sample -> frame conversion factor of the convolutional feature extractor
        frame_stride = np.prod(self.ssl.wav2vec2.config.conv_stride)

        target_repr = []
        for b in range(B):
            start_f = target_start[b] // frame_stride
            end_f = target_end[b] // frame_stride

            start_f = max(0, start_f)
            end_f = min(T, max(start_f + 1, end_f))

            target_frames = reps[b, start_f:end_f]
            context_frames = torch.cat(
                [reps[b, :start_f], reps[b, end_f:]],
                dim=0
            )

            q = target_frames.unsqueeze(0)             # (1, n_target, D)
            context = context_frames.unsqueeze(0)      # (1, n_context, D)

            # --- temporal mean pooling of the context, broadcast to the target ---
            if context.shape[1] == 0:
                # no context available (first utterance of a conversation)
                pooled = q.new_zeros((1, q.shape[1], D))
            else:
                pooled = context.mean(dim=1).unsqueeze(1).expand(-1, q.shape[1], -1)

            enriched = self.norm1(q + pooled)
            enriched = self.norm2(enriched + self.ffn(enriched))

            target_repr.append(enriched.mean(dim=1).squeeze(0))

        target_repr = torch.stack(target_repr, dim=0)
        return self.classifier(target_repr)
