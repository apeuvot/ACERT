# ACERT

Official implementation of **Enriching Speech Emotion Representations with
Conversational Context** (submitted to ICASSP 2027).

ACERT (Averaged Contextual Emotion Representation through Time) enriches the frame-level
representation of a target utterance with the conversation that precedes it: the context
frames are aggregated by temporal mean pooling and the resulting vector is added to every
target frame, before a feed-forward block and a classification head. The module sits on
top of any frame-level feature extractor; the paper uses
[HuBERT-large](https://huggingface.co/facebook/hubert-large-ll60k).

This repository contains what is needed to reproduce the experiments reported in the
paper: the baseline, ACERT on IEMOCAP / MELD / SAFE, the preliminary design experiments
and the ablation studies.


## Install

```bash
pip install -r requirements.txt
```

## Prepare the datasets

### IEMOCAP

Submit a [request](https://sail.usc.edu/iemocap/iemocap_release.htm) to get access to the
IEMOCAP dataset. Once granted, run the following, replacing `path/to/iemocap_dir` with the
directory containing `IEMOCAP_full_release`:

```bash
cd Dataset/IEMOCAP
python make_16k.py --iemocap_dir="path/to/iemocap_dir"
python gen_meta_label.py --iemocap_dir="path/to/iemocap_dir"
python generate_labels_sessionwise.py
python generate_conv_order_file.py --iemocap_dir="path/to/iemocap_dir"
cd ../..
```

This produces the 4 classes used in the paper (anger, happiness — merged with excitement —,
sadness, neutrality), one label file per session in `labels_sess/` (speaker-independent
5-fold cross-validation), and `conversations_orders.json`, which gives the chronological
order of the utterances of each conversation, both speakers mixed.

### MELD

Download the [MELD dataset](https://affective-meld.github.io/) and run, replacing
`path/to/meld_dir` with the directory containing `MELD.Raw`:

```bash
cd Dataset/MELD
python make_16k_and_generate_labels.py --meld_dir="path/to/meld_dir"
cd ../..
```

The standard train / dev / test splits of MELD are used, so `labels_sess/` holds a single
label file.

### SAFE

Submit a [request](https://clavel.wp.imt.fr/corpora/) to get access to the SAFE corpus,
then run:

```bash
cd Dataset/SAFE
python make_16k_and_generate_folds.py --safe_dir="path/to/safe_dir"
cd ../..
```

The 30 movies are randomly partitioned into 5 folds of 6 movies each (`--seed` controls
the partition). The published results use one such partition; a different seed gives
different folds and therefore slightly different numbers.

## Reproduce the results

Every script in `scripts/` runs `--num_exps 5` independent runs over all folds and writes
the mean and standard deviation to `results/<DATASET>/<run name>/results_Test.txt`.

```bash
bash scripts/run_acert_iemocap.sh
```

By default the model is downloaded from the Hub; set `MODEL_PATH` to use a local copy:

```bash
MODEL_PATH=/path/to/hubert-large-ll60k bash scripts/run_acert_iemocap.sh
```

## Credits & Acknowledgments

This repository is based on [FT-w2v2-ser](https://github.com/b04901014/FT-w2v2-ser), the
implementation of *Exploring wav2vec 2.0 fine-tuning for improved speech emotion
recognition* by Li-Wei Chen and Alexander Rudnicky. We thank the authors for making their
code available; it served as the foundation of the ACERT implementation.

This publication was made possible by the use of the CEA List FactoryIA supercomputer,
financially supported by the Ile-de-France Regional Council. This work was partially
supported by the ANR-23-PEIA0008 SHARP project in the context of the France 2030 program.

## Citation

If you use this code, or build on ACERT in your own work, please cite the paper:

```bibtex
@inproceedings{peuvot2027acert,
  title     = {Enriching Speech Emotion Representations with Conversational Context},
  author    = {Peuvot, Arthur and Besan\c{c}on, Romaric and de Chalendar, Ga\"el and
               Vieru, Bianca and Vasilescu, Ioana},
  booktitle = {},
  year      = {2027},
  pages     = {},
  doi       = {},
}
```

The empty fields will be filled once the paper is published. A preprint reference will be
added here as soon as one is available.
