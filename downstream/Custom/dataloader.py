import os
import re
import json
import random
from collections import Counter

import numpy as np
import soundfile as sf
import resampy
import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate


def load_audio_segment(dataname, max_samples=None, target_sr=16000, crop="random"):
    """Read at most ``max_samples`` samples, convert to mono and resample if needed.

    ``max_samples=None`` reads the whole file.

    crop : which part to keep when the file is longer than ``max_samples``.
        "random" : random start (default).
        "end"    : keep the END of the file. Used for past context, whose end is
                   adjacent to the target.
        "start"  : keep the BEGINNING of the file. Used for future context.
    Context segments are concatenated, so a random crop would glue the neighbour to
    the target with an arbitrary (and re-drawn at every epoch) time shift, breaking
    the continuity of the signal.

    Returns: wav (np.float32), sr
    """
    info = sf.info(dataname)

    if max_samples is not None and info.frames > max_samples:
        if crop == "random":
            start = np.random.randint(0, info.frames - max_samples)
        elif crop == "end":
            start = info.frames - max_samples
        elif crop == "start":
            start = 0
        else:
            raise ValueError(f"Unknown crop mode: {crop}")
        frames = max_samples
    else:
        start = 0
        frames = info.frames

    wav, sr = sf.read(dataname, start=start, frames=frames)

    # Mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
        print(f'Utterance {dataname} converted from stereo to mono')

    # Resample
    if sr != target_sr:
        wav = resampy.resample(wav, sr, target_sr)
        print(f'Utterance {dataname} converted from {sr}Hz to {target_sr}Hz')
        sr = target_sr

    wav = wav.astype(np.float32)
    return wav, sr


# =====================================================================================
#  With conversational context (ACERT)
# =====================================================================================

class CustomEmoContextDataset:
    """Builds the train / validation / test datasets used by ACERT.

    Each item is the target utterance concatenated with its conversational context,
    plus the sample indices delimiting the target inside that waveform.
    """

    def __init__(self, datadir, labeldir, dataset_name, maxseqlen, context_duration,
                 context_direction="past", context_speaker_dependent=False,
                 conv_order_file=None, speaker_diarization_file=None,
                 sample_rate=16000, context_source="recent", context_extra=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.context_source = context_source
        self.context_extra = context_extra
        self.maxseqlen = maxseqlen * sample_rate
        self.context_duration = context_duration
        self.context_direction = context_direction
        self.context_speaker_dependent = context_speaker_dependent
        self.conv_order_file = conv_order_file
        self.speaker_diarization_file = speaker_diarization_file

        assert context_duration is not None
        self.max_context_samples = int(context_duration * sample_rate)

        with open(labeldir, 'r') as f:
            self.label = json.load(f)  # {split: {wavname: emotion_label}}

        self.emoset = sorted(set(emo for split in self.label.values() for emo in split.values()))
        self.nemos = len(self.emoset)

        common = dict(
            dataset_name=dataset_name,
            maxseqlen=self.maxseqlen,
            max_context_samples=self.max_context_samples,
            context_direction=self.context_direction,
            context_speaker_dependent=self.context_speaker_dependent,
            conv_order_file=self.conv_order_file,
            speaker_diarization_file=self.speaker_diarization_file,
            context_source=self.context_source,
            context_extra=self.context_extra,
        )

        self.train_dataset = _CustomEmoContextDataset(
            datadir, self.label['Train'], self.emoset, 'training', **common)

        if 'Val' in self.label and self.label['Val']:
            self.val_dataset = _CustomEmoContextDataset(
                datadir, self.label['Val'], self.emoset, 'validation', **common)

        # several test splits are supported, the three corpora define a single one
        self.test_datasets = {}
        for test_key in self.label:
            if not test_key.lower().startswith("test"):
                continue
            test_labels = self.label[test_key]
            if not test_labels:
                continue
            self.test_datasets[test_key] = _CustomEmoContextDataset(
                datadir, test_labels, self.emoset, 'testing', **common)

    def get_collate_fn(self):
        return self.collate_temporal

    def collate_temporal(self, batch):
        B = len(batch)
        maxlen = max(x[1] for x in batch)

        audio = torch.zeros(B, maxlen)
        lengths = torch.zeros(B, dtype=torch.long)
        labels = torch.zeros(B, dtype=torch.long)
        t_start = torch.zeros(B, dtype=torch.long)
        t_end = torch.zeros(B, dtype=torch.long)

        for i, (wav, l, lab, s, e) in enumerate(batch):
            audio[i, :l] = torch.from_numpy(wav)
            lengths[i] = l
            labels[i] = lab
            t_start[i] = s
            t_end[i] = e

        return audio, lengths, labels, t_start, t_end


class _CustomEmoContextDataset(data.Dataset):
    def __init__(self, datadir, label, emoset, split, dataset_name="IEMOCAP", maxseqlen=10 * 16000,
                 max_context_samples=None, context_direction="past",
                 context_speaker_dependent=False, conv_order_file=None,
                 speaker_diarization_file=None, context_source="recent",
                 context_extra=False):
        super().__init__()
        self.maxseqlen = maxseqlen
        self.split = split
        self.label = label  # {wavname: emotion_label}
        self.emoset = emoset
        self.nemos = len(self.emoset)
        self.max_context_samples = max_context_samples
        self.context_direction = context_direction
        self.context_speaker_dependent = context_speaker_dependent
        self.context_source = context_source
        self.context_extra = context_extra
        self.session_to_convs = {}   # filled below (IEMOCAP, speaker-independent)
        self.length_cache = {}

        # Files of the split (the samples actually returned by the dataloader)
        self.datasetbase = list(self.label.keys())
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]
        self.datadir = datadir

        self.labeldict = {k: i for i, k in enumerate(self.emoset)}
        self.emos = Counter(self.label.values())

        # The context is built from every wav of the corpus, not only from the
        # labelled ones: 45% of IEMOCAP and 22% of SAFE carry an emotion outside the
        # label set of the task, and skipping them would leave holes in the
        # conversations. A conversation belongs entirely to one split, and the
        # context is only ever read as audio.
        self.context_files = sorted(
            f for f in os.listdir(datadir) if f.lower().endswith(".wav")
        )

        # Filename patterns: (conversation id, utterance index)
        if dataset_name.lower() == "iemocap":
            self.pattern = re.compile(
                r"^(Ses\d+[FM]_(?:impro|script)\d+[a-zA-Z]?(?:_[0-9a-zA-Z]+)?_[FM])(\d+)\.wav$"
            )
        elif dataset_name.lower() == "meld":
            # dialogue numbers restart in each split, so the split prefix belongs to
            # the conversation id
            self.pattern = re.compile(
                r"^((?:train|val|test)_dia\d+)_utt(\d+)\.wav$"
            )
        elif dataset_name.lower() == "safe":
            self.pattern = re.compile(
                r"^([a-zA-Z0-9]+_c\d+s\d+)_(\d{4})\.wav$"
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        self.speaker_diarization = None
        if speaker_diarization_file is not None:
            with open(speaker_diarization_file, "r") as f:
                self.speaker_diarization = json.load(f)

        # ------------------------------------------------------------
        # Speaker-dependent context for IEMOCAP.
        # MELD and SAFE always go through this branch: their conversation id comes
        # from the filename and no speaker information is available.
        # ------------------------------------------------------------
        if (self.context_speaker_dependent and dataset_name.lower() == "iemocap") or dataset_name.lower() in ["meld", "safe"]:

            # --------------------------------------------------------
            # CASE 1: speaker diarization file provided (IEMOCAP only)
            # --------------------------------------------------------
            if dataset_name.lower() == "iemocap" and self.speaker_diarization is not None:

                self.groups = {}

                for conv_id, speaker_lists in self.speaker_diarization.items():
                    for utt_list in speaker_lists:

                        if len(utt_list) == 0:
                            continue

                        # Speaker id comes from utterance name (…_M000 / …_F000)
                        first_utt = utt_list[0]
                        speaker = first_utt.split("_")[-1][0]  # 'M' or 'F'

                        group_id = f"{conv_id}_{speaker}"
                        self.groups.setdefault(group_id, [])

                        for idx, utt_name in enumerate(utt_list):
                            fname = utt_name + ".wav"

                            if fname not in self.context_files:
                                continue

                            self.groups[group_id].append((idx, fname))

                # Mapping fname -> (group_id, utt_idx)
                self.file_to_conv = {
                    fname: (group_id, utt_idx)
                    for group_id, utt_list in self.groups.items()
                    for utt_idx, fname in utt_list
                }

            # --------------------------------------------------------
            # CASE 2: grouping based on the filename pattern
            # --------------------------------------------------------
            else:
                self.groups = {}
                for fname in self.context_files:
                    m = self.pattern.match(fname)
                    if not m:
                        raise ValueError(f"Unexpected filename: {fname}")
                    conv_id, utt_idx = m.groups()
                    self.groups.setdefault(conv_id, []).append((utt_idx, fname))

                for conv_id in self.groups:
                    self.groups[conv_id].sort(key=lambda x: self.utt_key(x[0]))

                # Mapping fname -> (conv_id, utt_idx)
                self.file_to_conv = {
                    fname: (conv_id, utt_idx)
                    for conv_id, utt_list in self.groups.items()
                    for utt_idx, fname in utt_list
                }

        # ------------------------------------------------------------
        # Speaker-independent context for IEMOCAP (default of the paper):
        # the conversation order comes from the transcription files.
        # ------------------------------------------------------------
        elif not self.context_speaker_dependent and dataset_name.lower() == "iemocap":

            if conv_order_file is None:
                raise ValueError("conv_order_file is needed when context_speaker_dependent=False")

            with open(conv_order_file, "r", encoding="utf-8") as f:
                conv_orders = json.load(f)

            self.groups = {}
            for conv_name, utt_list in conv_orders.items():
                clean_list = [
                    fname for fname in (u + ".wav" for u in utt_list)
                    if fname in self.context_files
                ]
                self.groups[conv_name] = [(i, fname) for i, fname in enumerate(clean_list)]

            for conv_name in self.groups:
                self.groups[conv_name].sort(key=lambda x: x[0])

            # Mapping fname -> (conv_name, order_idx)
            self.file_to_conv = {
                fname: (conv_name, utt_idx)
                for conv_name, utt_list in self.groups.items()
                for utt_idx, fname in utt_list
            }

            # session ('Ses0X') -> conversations, for the same-session ablations
            for conv_name in self.groups:
                self.session_to_convs.setdefault(conv_name[:5], []).append(conv_name)
        else:
            raise ValueError("Dataset not supported")

        # Print statistics
        print(f'Statistics of {self.split} splits:')
        print('----Involved Emotions----')
        for k, v in self.emos.items():
            print(f'{k}: {v} examples')
        print(f'Total : {len(self.dataset)} examples\n')

        if self.context_source in ("distant_same_session", "distant_scattered_session") and not self.session_to_convs:
            raise ValueError(f"context_source='{self.context_source}' is only supported for IEMOCAP "
                             "in speaker-independent mode (self.session_to_convs is empty).")

    def utt_key(self, utt):
        parts = utt.split('_')
        base = int(parts[0])
        sub = int(parts[1]) if len(parts) > 1 else -1
        return (base, sub)

    def __len__(self):
        return len(self.datasetbase)

    def __getitem__(self, i):
        fname = self.datasetbase[i]
        label = self.labeldict[self.label[fname]]

        conv_id, utt_idx = self.file_to_conv[fname]
        utt_list = self.groups[conv_id]
        base_dir = self.datadir

        # ---------------------------------------------
        # Load the target utterance
        # ---------------------------------------------
        wav_path = os.path.join(base_dir, fname)
        # Target cap: in "extra" mode the target goes up to maxseqlen (independently
        # of context_duration); otherwise it is capped by the total budget.
        target_cap = self.maxseqlen if self.context_extra else self.max_context_samples
        target_wav, _sr = load_audio_segment(wav_path, max_samples=target_cap)
        target_len = len(target_wav)

        # target alone already filling the budget; in extra mode context is always
        # added on top, so there is nothing to short-circuit
        if not self.context_extra and target_len >= self.max_context_samples:
            self.length_cache[fname] = target_len
            return target_wav, target_len, label, 0, target_len

        # ---------------------------------------------
        # Position of the target inside the conversation
        # ---------------------------------------------
        pos_cible = next(
            (i for i, (conv_idx, _) in enumerate(utt_list) if conv_idx == utt_idx),
            None
        )
        if pos_cible is None:
            raise ValueError(f"utt_idx {utt_idx} not found in utt_list for {fname}")

        # ---------------------------------------------
        # Remaining context budget
        # ---------------------------------------------
        # extra mode adds the whole context_duration on top of the target, otherwise
        # the context gets what is left of the total budget
        remaining = self.max_context_samples if self.context_extra else (self.max_context_samples - target_len)

        if self.context_direction == "past":
            past_budget = remaining
            future_budget = 0
        elif self.context_direction == "past_and_future":
            past_budget = remaining // 2
            future_budget = remaining - past_budget  # handles odd values
        else:
            raise ValueError(f"Unknown context_direction: {self.context_direction}")

        # ---------------------------------------------
        # Load the past context
        # ---------------------------------------------
        past_segments = []
        past_len = 0

        if self.context_source in ("distant_same_session", "distant_scattered_session"):
            # Ablation: same two speakers and same recording conditions (same IEMOCAP
            # session) but drawn from ANOTHER conversation, so no conversational or
            # emotional continuity with the target. Same time budget as usual.
            conv_name = self.file_to_conv[fname][0]
            other_convs = [c for c in self.session_to_convs.get(conv_name[:5], [])
                           if c != conv_name]
            rng = random.Random(fname)   # deterministic draw -> reproducible

            if self.context_source == "distant_same_session":
                # Continuous window: the end of ONE distant conversation.
                if other_convs:
                    distant_conv = rng.choice(other_convs)
                    distant_utts = [f for _, f in self.groups[distant_conv]]
                    neighbor_iter = list(reversed(distant_utts))
                else:
                    neighbor_iter = []
            else:
                # Scattered: utterances drawn at random from all the other
                # conversations of the session, no temporal continuity at all.
                pool = [f for c in other_convs for _, f in self.groups[c]]
                rng.shuffle(pool)
                neighbor_iter = pool

            for neighbor_fname in neighbor_iter:
                if past_len >= past_budget:
                    break
                max_samples = past_budget - past_len
                wav, _sr = load_audio_segment(os.path.join(base_dir, neighbor_fname),
                                              max_samples=max_samples, crop="end")
                past_segments.insert(0, wav)
                past_len += len(wav)
        else:
            for idx in range(pos_cible - 1, -1, -1):
                if past_len >= past_budget:
                    break

                neighbor_fname = utt_list[idx][1]
                max_samples = past_budget - past_len
                wav, _sr = load_audio_segment(os.path.join(base_dir, neighbor_fname),
                                              max_samples=max_samples, crop="end")

                past_segments.insert(0, wav)  # chronological order
                past_len += len(wav)

        # ---------------------------------------------
        # Load the future context
        # ---------------------------------------------
        future_segments = []
        future_len = 0

        for idx in range(pos_cible + 1, len(utt_list)):
            if future_len >= future_budget:
                break

            neighbor_fname = utt_list[idx][1]
            max_samples = future_budget - future_len
            wav, _sr = load_audio_segment(os.path.join(base_dir, neighbor_fname),
                                          max_samples=max_samples, crop="start")

            future_segments.append(wav)
            future_len += len(wav)

        # ---------------------------------------------
        # Assemble
        # ---------------------------------------------
        audio = np.concatenate(past_segments + [target_wav] + future_segments)
        target_start = past_len
        target_end = past_len + target_len

        self.length_cache[fname] = len(audio)
        return audio, len(audio), label, target_start, target_end


# =====================================================================================
#  Without conversational context (baseline)
# =====================================================================================

class CustomEmoDataset:
    def __init__(self, datadir, labeldir, maxseqlen):
        super().__init__()
        self.maxseqlen = maxseqlen * 16000  # Assume sample rate of 16000
        with open(labeldir, 'r') as f:
            self.label = json.load(f)  # {split: {wavname: emotion_label}}

        self.emoset = sorted(set(emo for split in self.label.values() for emo in split.values()))
        self.nemos = len(self.emoset)

        self.train_dataset = _CustomEmoDataset(datadir, self.label['Train'], self.emoset,
                                               'training', maxseqlen=self.maxseqlen)
        if 'Val' in self.label and self.label['Val']:
            self.val_dataset = _CustomEmoDataset(datadir, self.label['Val'], self.emoset,
                                                 'validation', maxseqlen=self.maxseqlen)

        self.test_datasets = {}
        for test_key in self.label:
            if not test_key.lower().startswith("test"):
                continue
            test_labels = self.label[test_key]
            if not test_labels:
                continue
            self.test_datasets[test_key] = _CustomEmoDataset(datadir, test_labels, self.emoset,
                                                             'testing', maxseqlen=self.maxseqlen)

    def get_collate_fn(self):
        return self.seqCollate

    def seqCollate(self, batch):
        getlen = lambda x: x[0].shape[0]
        max_seqlen = max(map(getlen, batch))
        target_seqlen = min(self.maxseqlen, max_seqlen)

        def trunc(x):
            x = list(x)
            if x[0].shape[0] >= target_seqlen:
                x[0] = x[0][:target_seqlen]
                output_length = target_seqlen
            else:
                output_length = x[0].shape[0]
                over = target_seqlen - x[0].shape[0]
                x[0] = np.pad(x[0], [0, over])
            return (x[0], output_length, x[1])

        batch = list(map(trunc, batch))
        return default_collate(batch)


class _CustomEmoDataset(data.Dataset):
    """Utterance-level dataset, without conversational context.

    The whole utterance is read here; the truncation to ``maxseqlen`` is done by
    :meth:`CustomEmoDataset.seqCollate`, which keeps the first samples.
    """

    def __init__(self, datadir, label, emoset, split, maxseqlen=10 * 16000):
        super().__init__()
        self.maxseqlen = maxseqlen
        self.split = split
        self.label = label
        self.emoset = emoset
        self.nemos = len(self.emoset)
        self.datasetbase = list(self.label.keys())
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]

        self.labeldict = {k: i for i, k in enumerate(self.emoset)}
        self.emos = Counter(self.label.values())

        print(f'Statistics of {self.split} splits:')
        print('----Involved Emotions----')
        for k, v in self.emos.items():
            print(f'{k}: {v} examples')
        print(f'Total : {len(self.dataset)} examples\n')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        wav, _sr = load_audio_segment(self.dataset[i], max_samples=None)
        label = self.labeldict[self.label[self.datasetbase[i]]]
        return wav, label
