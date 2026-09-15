"""Fine-tune an SSL encoder for Speech Emotion Recognition, with or without ACERT.

Runs ``--num_exps`` independent runs over every fold found in ``--labeldir`` and
reports the mean and standard deviation of UA, WA, macro-F1 and weighted-F1.
See the README for the exact command lines reproducing the paper's table.
"""

import os
import argparse

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from downstream.Custom.trainer import DownstreamGeneral
from utils.helper_funcs import gather_confusion_matrix
from utils.outputlib import WriteConfusionSeaborn, compute_metrics_per_class, print_metrics


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

# --- model / optimisation ---
parser.add_argument('--model_path', type=str, required=True,
                    help="HuggingFace id or local path of the SSL encoder "
                         "(the paper uses facebook/hubert-large-ll60k)")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--maxseqlen', type=int, default=10,
                    help="maximum input duration in seconds for the target utterance")
parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--training_strategy', type=str, choices=["DDP", "Deepspeed", None], default=None)

# --- data ---
parser.add_argument('--datadir', type=str, required=True,
                    help="directory holding the 16 kHz wav files")
parser.add_argument('--dataset_name', type=str, required=True, choices=['IEMOCAP', 'MELD', 'SAFE'])
parser.add_argument('--labeldir', type=str, required=True,
                    help="directory holding one json label file per fold")

# --- experiment ---
parser.add_argument('--saving_path', type=str, default='downstream/checkpoints/custom')
parser.add_argument('--save_top_k', type=int, default=1)
parser.add_argument('--num_exps', type=int, default=1,
                    help="number of independent runs to average over")

# --- conversational context (ACERT) ---
parser.add_argument('--context_duration', type=float, default=None,
                    help="Enables ACERT: duration in seconds of the audio window fed to "
                         "the model. By default this is the TOTAL window (target utterance "
                         "+ context): the target is truncated to at most context_duration "
                         "and the context fills the remainder. This is the setting used for "
                         "every result reported in the paper. Pass --context_extra to "
                         "instead add context_duration seconds of context on top of the "
                         "target. Without this argument the model is the utterance-level "
                         "baseline and every argument below is ignored.")
parser.add_argument('--context_extra', action='store_true',
                    help="If set, context_duration is the PURE context added on top of the "
                         "target: the context budget = context_duration, and the target is "
                         "loaded separately up to maxseqlen. Not used for the published "
                         "results.")
parser.add_argument('--context_direction', type=str, choices=['past', 'past_and_future'],
                    default='past',
                    help="Side of the target the context is taken from. 'past' = only the "
                         "utterances preceding the target (default, used for the main "
                         "results); 'past_and_future' splits the budget evenly before and "
                         "after the target.")
parser.add_argument('--context_source', type=str,
                    choices=['recent', 'distant_same_session', 'distant_scattered_session'],
                    default='recent',
                    help="'recent' = the seconds preceding the target (default). "
                         "'distant_same_session' = a window taken from the end of ANOTHER "
                         "conversation of the same IEMOCAP session (same two speakers, same "
                         "recording conditions, no conversational continuity). "
                         "'distant_scattered_session' = the budget filled with utterances "
                         "drawn at random from the session's other conversations. The two "
                         "distant modes are the ablation studies of the paper and are only "
                         "available for IEMOCAP in speaker-independent mode.")
parser.add_argument('--context_speaker_dependent', action='store_true',
                    help="IEMOCAP only: restrict the context to the utterances of the same "
                         "speaker as the target. Without --speaker_diarization_file the "
                         "speaker comes from the ground-truth _F/_M tag of the filename; "
                         "with it, from the predicted diarization clusters.")
parser.add_argument('--speaker_diarization_file', type=str, default=None,
                    help="json produced by speaker_diarization.py, mapping each conversation "
                         "to its predicted speaker groups. Used with "
                         "--context_speaker_dependent.")
parser.add_argument('--conv_order_file', type=str, default=None,
                    help="IEMOCAP only: json giving the chronological order of the "
                         "utterances of each conversation. Required unless "
                         "--context_speaker_dependent is set.")

args = parser.parse_args()
hparams = args

print(f'Dataset name : {hparams.dataset_name}')
print(f'Model : {hparams.model_path}')
if hparams.context_duration is None:
    print('Conversational context : none (utterance-level baseline)')
else:
    print(f'Context duration : {hparams.context_duration}')
    print(f'Context source : {hparams.context_source}')
    print(f'Context direction : {hparams.context_direction}')
    print(f'Context extra : {hparams.context_extra}')
    print(f'Context speaker dependent : {hparams.context_speaker_dependent}')
print(f'Learning rate : {hparams.lr}')

# Automatically detect every available GPU
nb_gpus = torch.cuda.device_count()

os.makedirs(hparams.saving_path, exist_ok=True)

foldlabels = sorted(os.listdir(hparams.labeldir))
for foldlabel in foldlabels:
    assert foldlabel[-5:] == '.json'
nfolds = len(foldlabels)

all_test_metrics = {}
all_test_confusions = {}

for exp in range(args.num_exps):
    for ifold, foldlabel in enumerate(foldlabels):
        print(f"Running experiment {exp + 1} / {args.num_exps}, fold {ifold + 1} / {nfolds}...")
        print(f'Running on file {foldlabel}')

        hparams.labelpath = os.path.join(hparams.labeldir, foldlabel)
        model = DownstreamGeneral(hparams)
        has_val = hasattr(model, 'valid_metrics')

        checkpoint_callback = ModelCheckpoint(
            dirpath=hparams.saving_path,
            filename='{epoch:02d}-{valid_loss:.3f}-{valid_UAR:.5f}' if has_val else None,
            save_top_k=args.save_top_k if has_val else 0,
            verbose=True,
            save_weights_only=True,
            monitor='valid_UAR' if has_val else None,
            mode='max'
        )

        # Metric accumulators, one entry per test split
        if exp == 0 and ifold == 0:
            for test_key in model.dataset.test_datasets:
                # [UA, WA, macroF1, weightedF1]
                all_test_metrics[test_key] = np.zeros((4, args.num_exps, nfolds))
                n_emos = len(model.dataset.emoset)
                all_test_confusions[test_key] = np.zeros((n_emos, n_emos), dtype=float)

        if args.training_strategy == "DDP" and nb_gpus > 1:
            strategy = DDPStrategy(find_unused_parameters=True)
        elif args.training_strategy == "Deepspeed" and nb_gpus > 1:
            strategy = DeepSpeedStrategy(stage=2)
        else:
            strategy = "auto"

        trainer = Trainer(
            precision=args.precision,
            devices=nb_gpus if nb_gpus > 0 else 1,
            strategy=strategy,
            callbacks=[checkpoint_callback] if has_val else None,
            enable_checkpointing=has_val,
            max_epochs=hparams.max_epochs,
            check_val_every_n_epoch=1 if hasattr(model.dataset, 'val_dataset') else None,
            num_sanity_val_steps=2 if has_val else 0,
            logger=False,
            accumulate_grad_batches=4,
        )

        trainer.fit(model)

        for test_key, loader in model.test_dataloaders().items():
            model.current_test_key = test_key
            trainer.test(model, dataloaders=loader)

            met = model.test_metrics[test_key]
            if torch.distributed.is_initialized():
                met.m = gather_confusion_matrix(met.m)

            all_test_metrics[test_key][:, exp, ifold] = np.array([
                met.uar * 100, met.war * 100, met.macroF1 * 100, met.weightedF1 * 100
            ])
            all_test_confusions[test_key] += met.m


for test_key, metrics in all_test_metrics.items():
    confusion = all_test_confusions[test_key]
    class_names = model.dataset.emoset
    saving_file = os.path.join(args.saving_path, f'results_{test_key}.txt')

    print(f"\n=== SUMMARY for test set {test_key} ===\n")

    outputstr = f"+++ SUMMARY for {test_key} +++\n"
    for nm, metric in zip(('UAR [%]', 'WAR [%]', 'macroF1 [%]', 'weightedF1 [%]'), metrics):
        outputstr += f"Mean {nm}: {np.mean(metric):.2f}\n"
        outputstr += f"Fold Std. {nm}: {np.mean(np.std(metric, 1)):.2f}\n"
        outputstr += f"Fold Median {nm}: {np.mean(np.median(metric, 1)):.2f}\n"
        outputstr += f"Run Std. {nm}: {np.std(np.mean(metric, 1)):.2f}\n"
        outputstr += f"Run Median {nm}: {np.median(np.mean(metric, 1)):.2f}\n"
        outputstr += "-------------------\n"

    fold_str = f"+++ RESULTS PER FOLD for {test_key} +++\n"
    for ifold, foldlabel in enumerate(foldlabels):
        fold_str += f"Fold {ifold + 1} ({foldlabel}):\n"
        uar, war, macroF1, weightedF1 = metrics[:, :, ifold].mean(axis=1)
        fold_str += f"  UAR [%]: {uar:.2f}\n"
        fold_str += f"  WAR [%]: {war:.2f}\n"
        fold_str += f"  macroF1 [%]: {macroF1:.2f}\n"
        fold_str += f"  weightedF1 [%]: {weightedF1:.2f}\n"
        fold_str += "-------------------\n"

    full_output = outputstr + fold_str
    print(full_output)

    results_dict = compute_metrics_per_class(confusion, metrics, class_names, excluded_labels=[])
    print_metrics(results_dict, file=saving_file, extra_text=full_output)

    WriteConfusionSeaborn(
        confusion,
        class_names,
        os.path.join(args.saving_path, f'confusion_matrice_{test_key}.png')
    )


# Save the arguments of the run next to its results
with open(os.path.join(args.saving_path, 'arguments.txt'), 'w') as f:
    for key, value in vars(args).items():
        f.write(f"{key}: {value}\n")
