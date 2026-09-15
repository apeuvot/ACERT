import argparse
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from pytorch_lightning import LightningModule

from .dataloader import CustomEmoDataset, CustomEmoContextDataset
from modules.heads import BaselineHead, ACERT
from utils.metrics import ConfusionMetrics


class DurationBucketSampler(Sampler):
    """
    Sampler for a DataLoader that groups samples by duration before forming batches.

    Purpose:
        Reduce unnecessary padding for variable-length audio data by grouping
        samples with similar durations into the same batch.

    Parameters:
        dataset (_CustomEmoContextDataset): dataset containing audio samples.
        batch_size (int): Number of samples per batch.
        bucket_size_sec (float): Size of each bucket in seconds. Samples are grouped
            into buckets based on multiples of bucket_size_sec.
        sample_rate (int): Audio sampling rate, used to convert seconds into number of samples.
        drop_last (bool): If True, the last incomplete batch in each bucket is dropped.
    """

    def __init__(self, dataset, batch_size, bucket_size_sec=10.0, sample_rate=16000, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = int(bucket_size_sec * sample_rate)
        self.drop_last = drop_last

        buckets = defaultdict(list)

        for i, fname in enumerate(dataset.datasetbase):
            if fname in dataset.length_cache:
                L = dataset.length_cache[fname]
            else:
                L = dataset.max_context_samples  # safe fallback

            bucket_id = L // self.bucket_size
            buckets[bucket_id].append(i)

        self.buckets = list(buckets.values())

        for b in self.buckets:
            random.shuffle(b)

        self.batches = []
        for b in self.buckets:
            for i in range(0, len(b), batch_size):
                batch = b[i:i + batch_size]
                if len(batch) == batch_size or not drop_last:
                    self.batches.append(batch)

        random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class DownstreamGeneral(LightningModule):
    """Fine-tuning of the SSL encoder for SER, with or without conversational context.

    A ``context_duration`` selects ACERT; without one the model is the
    utterance-level baseline.
    """

    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hp = hparams

        self.use_context = self.hp.context_duration is not None

        if self.use_context:
            self.dataset = CustomEmoContextDataset(
                self.hp.datadir,
                self.hp.labelpath,
                self.hp.dataset_name,
                maxseqlen=self.hp.maxseqlen,
                context_duration=self.hp.context_duration,
                context_direction=self.hp.context_direction,
                context_speaker_dependent=self.hp.context_speaker_dependent,
                conv_order_file=self.hp.conv_order_file,
                speaker_diarization_file=self.hp.speaker_diarization_file,
                context_source=self.hp.context_source,
                context_extra=self.hp.context_extra,
            )
            self.model = ACERT(
                n_outputs=self.dataset.nemos,
                model_path=self.hp.model_path,
            )
        else:
            self.dataset = CustomEmoDataset(
                self.hp.datadir,
                self.hp.labelpath,
                maxseqlen=self.hp.maxseqlen,
            )
            self.model = BaselineHead(
                n_outputs=self.dataset.nemos,
                model_path=self.hp.model_path,
            )

        # Cross-entropy weighted by the inverse prior of each class
        counter = self.dataset.train_dataset.emos
        weights = torch.tensor([counter[c] for c in self.dataset.emoset]).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        print(f"Weigh losses by prior distribution of each class: {weights}.")
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        if hasattr(self.dataset, 'val_dataset'):
            self.valid_metrics = ConfusionMetrics(self.dataset.nemos)

        self.test_metrics = {
            test_key: ConfusionMetrics(self.dataset.nemos)
            for test_key in self.dataset.test_datasets
        }

    # ------------------------------------------------------------------ optimizer

    def configure_optimizers(self):
        return optim.Adam(self.model.trainable_params(), lr=self.hp.lr)

    # ---------------------------------------------------------------- dataloaders

    def train_dataloader(self):
        """
        Return the training DataLoader.
        - If a conversational context is used and a single GPU is available, use
          DurationBucketSampler (does not work with DDP).
        - Otherwise, use a standard DataLoader with shuffle / DistributedSampler.
        """
        nb_gpus = self.trainer.num_devices if hasattr(self.trainer, "num_devices") else 1
        use_distributed = nb_gpus > 1

        if self.use_context and not use_distributed:
            batch_sampler = DurationBucketSampler(
                dataset=self.dataset.train_dataset,
                batch_size=self.hp.batch_size,
                bucket_size_sec=10.0,
                sample_rate=self.dataset.sample_rate,
                drop_last=True,
            )

            return DataLoader(
                dataset=self.dataset.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.dataset.get_collate_fn(),
                num_workers=self.hp.nworkers,
            )

        sampler = (
            torch.utils.data.distributed.DistributedSampler(self.dataset.train_dataset)
            if use_distributed else None
        )

        return DataLoader(
            dataset=self.dataset.train_dataset,
            batch_size=self.hp.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=self.dataset.get_collate_fn(),
            num_workers=self.hp.nworkers,
            drop_last=True,  # safety for DDP
        )

    def val_dataloader(self):
        # no val_dataset -> no validation
        if not hasattr(self.dataset, "val_dataset") or len(self.dataset.val_dataset) == 0:
            return []

        use_distributed = (
            hasattr(self.trainer, "strategy")
            and self.trainer.num_devices is not None
            and self.trainer.num_devices > 1
        )

        sampler = (
            torch.utils.data.distributed.DistributedSampler(self.dataset.val_dataset, shuffle=False)
            if use_distributed
            else None
        )

        return DataLoader(
            dataset=self.dataset.val_dataset,
            collate_fn=self.dataset.get_collate_fn(),
            batch_size=self.hp.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.hp.nworkers,
            drop_last=True,  # safety for DDP
        )

    def test_dataloaders(self):
        use_distributed = (
            hasattr(self.trainer, "strategy")
            and self.trainer.num_devices is not None
            and self.trainer.num_devices > 1
        )

        loaders = {}
        for test_key, test_dataset in self.dataset.test_datasets.items():
            sampler = (
                torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
                if use_distributed
                else None
            )

            loaders[test_key] = DataLoader(
                dataset=test_dataset,
                batch_size=self.hp.batch_size,
                num_workers=self.hp.nworkers,
                collate_fn=self.dataset.get_collate_fn(),
                sampler=sampler,
                shuffle=False,
                drop_last=True,
            )

        return loaders

    # ---------------------------------------------------------------------- steps

    def _forward_batch(self, batch):
        if self.use_context:
            audio, lengths, label, target_start, target_end = batch
            return self.model(audio, lengths, target_start, target_end), label

        utterances, lengths, label = batch
        return self.model(utterances, lengths), label

    def training_step(self, batch, batch_idx):
        pout, label = self._forward_batch(batch)
        loss = self.criterion(pout, label)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pout, label = self._forward_batch(batch)

        loss = self.criterion(pout, label)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for l, p in zip(label, pout):
            self.valid_metrics.fit(int(l), int(p.argmax()))

    def test_step(self, batch, batch_idx):
        pout, label = self._forward_batch(batch)

        met = self.test_metrics[self.current_test_key]
        for l, p in zip(label, pout):
            met.fit(int(l), int(p.argmax()))

    # -------------------------------------------------------------- epoch reports

    def on_validation_epoch_end(self):
        print(f"Validation UAR: {self.valid_metrics.uar:.4f}")
        self.log('valid_UAR', self.valid_metrics.uar)
        self.log('valid_WAR', self.valid_metrics.war)
        self.log('valid_macroF1', self.valid_metrics.macroF1)
        self.log('valid_weightedF1', self.valid_metrics.weightedF1)
        self.valid_metrics.clear()

    def on_test_epoch_end(self):
        """Report metrics for the test set that just finished."""
        test_key = self.current_test_key
        met = self.test_metrics[test_key]

        self.log(f'test_{test_key}_UAR', met.uar, logger=True)
        self.log(f'test_{test_key}_WAR', met.war, logger=True)
        self.log(f'test_{test_key}_macroF1', met.macroF1, logger=True)
        self.log(f'test_{test_key}_weightedF1', met.weightedF1, logger=True)

        print(f"""\n++++ Classification Metrics [{test_key}] ++++
                  UAR: {met.uar:.4f}
                  WAR: {met.war:.4f}
                  macroF1: {met.macroF1:.4f}
                  weightedF1: {met.weightedF1:.4f}""")
