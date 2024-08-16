# Testing the Features by Training the Decoder


The HMM decoder is trained on the features extracted from the BrainGPT model. The features are extracted from the BrainGPT model and then used to train the decoder. The decoder is then tested on the testing dataset to evaluate its performance.

## Testing Data
We are using Rehabilitation day 9 (4th of December 2023).
But you can use any other session where a decoder was trained and tested.

## What are .dt5 files?
During the session, Wisci software records the data and saves it in .smr files. Then, we launch the ABSD software that works with the decoder - via this app you can train the decoder and run inference. It generates the .dt5 files.

The .dt5 files therefore contain the session data that were used to train the decoder (online) during the session. Now, the goal is to take the data and simulate the training offline.

The .dt5 files contain the following fields:
- X ... 4d tensor (10, 24, 32, timestamps by 100ms)
- xLatent ... (768, timestamps by 100ms) .. only when the BrainGPT was run during the session. But we don't want to use this because if we are testing a diferent model than the one from the session, we need to compute the latents from our model. This is done with `quick_latent_extractor.ipynb` or `quick_latent_extractor.py`. Check bellow the steps.
- yDesired ... 13xtimestamps by 100ms .. onehot encoding of the prediction
- events ... string version of yDesired
- isUpdating ... if decoder is updating at that timepoint - so the decoder is trained when it is 1
- probaPredicted ... the realtime probability that the decoder
- triggers ... there’s LED (hopefully)
- raw ... raw ecog = like .smr … could be used to synchronize with .smr file. It’s sampled with the same freq

## Steps

1. Download all the dt5 files (using `copy_only_dt5_files.py`)
2. I use `test_dt5.m` where I call functions from loadDataM.m and save the ‘X’ data into a mat file. All the .dt5 files from a session are concatenated (the files are usually called `_01of03.dt5`, `_02of03.dt5`, `_03of03.dt5`) and saved into one file.
3. Then use `quick_latent_extractor.py` (or `quick_latent_extractor.ipynb` for better flexibility and less automation with the code) to generate new BrainGPT features (with our custom model we are testing). It is generated from the file saved in the last step. The features are then saved into a .mat file.
4. Finally, the script `wavelet_vs_vanilla_vs_multimodal.m` can be used to run a bunch of different analyses.
 - It uses the `ModelRecomputation` tool from Clinatec to train the decoder offline
 - The script tries different amounts of training dataset to train the decoder
 - Then features from different models are compared by training the decoder on them (also trained on different amounts of data). We try wavelet features, BrainGPT features without any modification (we call it Vanilla BrainGPT), and the multimodal BrainGPT features. The F1 scores are compared.
 - Comparison of different multimodalities is performed, too (the values are precomputed and hardcoded in the script, so that we don't need to recompute it again).
 - Given a cue, the probabilities are visualized in a plot
 - Confusion matrix plot is generated
 - And the T-SNE plot is generated, too