# gridspace-stanford-harper-valley
The Gridspace-Stanford Harper Valley speech dataset. Created in support of CS224S. This folder contains the code for experiments listed in the paper.

## Experiment Overview 
We include two sets of experiments:
    - The first set of experiments fits supervised algorithms for automatic speech recognition on the HarperValleyBank corpus. These algorithms include CTC, Listen-Attend-Spell, and a joint "multi-task" objective. We fit each of these with additional auxiliary task objectives such as predicting speaker identity, dialog actions, and more. 
    - The second set of experiments fits a suite of unsupervised representation learning algorithms, called contrastive algorithms, on the Librispeech corpus to extract vector representations on top of log Mel spectrograms. We then compare the effectiveness of individual contrastive objectives by linear evaluation where a logistic regression model to trained on top of representation vectors to predict speaker identity, dialog actions, and more. A better representation would indicate higher accuracy on these transfer tasks. We additionally compare to Wav2Vec, a well-known baseline.

## Installation and Dependencies
We include a yaml file with the conda environment used to run the experiments. Notably, we use the PyTorch Lightning framework and the Weights and Biases library for visualization. The experiments were run PyTorch 1.6.0. 

## Usage Instructions

The code requires the HarperValleyBank dataset, which can be found in the repository (one folder up), but it also requires the [Librispeech dataset](http://www.openslr.org/12/). You will need to download the 100 hour, 360 hour, and 500 hour splits into the same folder. Additionally, pretrained weights for the Wav2Vec-1.0 and Wav2Vec-2.0 baselines can be found in a [public repository](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec).

The experiment code is set up as a package and as such, before using it, the user needs to run the following: 
```
source init_env.sh
```
so that proper paths are accessible locally. If you get an error reporting that certain local imports are missing, try running the command above. 

The master command for all experiments is:
```
python scripts/run.py <PATH_TO_CONFIG>
```
Thus, the kind of experiment you wish to run depends on the config file. The config files are split into two folders: `config/asr` and `config/transfer`, the former being for the speech recognition experiments and the latter being for the representation learning experiments. In `config/transfer`, you will find the config files for the Wav2Vec baselines as well as the folder `config/transfer/contrastive` that contains config files for the different splits (e.g. representations learned on 100 hours of Librispeech versus all 960 hours). Further, some of the folders have a `_spectral` suffix which indicates that augmentations will be applied to the log Mel spectrogram rather than the waveforms themselves. Finally, each of these folders (e.g. `config/transfer/contrastive/100hr`) contain config files for the varous contrastive algorithms (IR, LA, MoCo, and SimCLR), as well as a subfolder named `harpervalley` that contain configs for transfer learning (after a representation has been trained). 

**A final note**: all of the config files have some fields that have `null` as their default value that the **user must replace**. Critically, the `exp_base` field represents where to store model checkpoints and the like. It should be replaced with a path to a folder in your system. Simiarly, the fields `librispeech_root` or `harpervalley_root` must be replaced by the locations that they have been downloaded to. If these fields are not replaced, you will receive errors upon execution.

## Questions? 
Please either raise Github issues or email <wumike@stanford.edu> with questions and concerns! We will try to respond as quickly as we can.