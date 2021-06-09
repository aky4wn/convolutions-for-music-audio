# convolutions-for-music-audio

Code for "Towards Explainable Convolutional Features for Music Audio Modeling", https://arxiv.org/abs/2106.00110.

Code reproduces NSynth results from the main paper and supplementary materials, as this data is publically available and the results are directly reproducible.  Code for the Composer and Beethoven dataset results is analogous; this data is not directly publically available, but results are reproducible with the same input data.  All results are as IPython notebooks and additional details about the code are found in S2.6 of the Supplementary Materials.

## Data Processing

- ``NSynth-Pre-Process.ipynb``: download NSynth data from https://magenta.tensorflow.org/datasets/nsynth, pre-process into mel-spectrograms, calculated hand-crafted and wavelet features, perform EDA.
- ``NSynth-Deep-Features.ipynb``: train convolutional architectures and extract deep features (to predict Instrument Family).  Experiments with the number of channels for the deep models also included.
- ``NSynth-Deep-Features-Transfer-Pitch.ipynb``: train convolutional architectures (Regular and Deformable) and extract deep features to predict Note Pitch.
- ``NSynth-Deep-Features-Transfer-Velocity.ipynb``: train convolutional architectures (Regular and Deformable) and extract deep features to predict Note Velocity.

## Experiments

### Classification

- ``NSynth-Classify.ipynb``: Classify deep features from the last convolutional layer and all hand-crafted features. Perform classification for main Instrument Family task and linear regression for related tasks of predicting Note Pitch and Note Velocity.
- ``NSynth-Concat-Features.ipynb``: Concatenate top hand-crafted features to the deep features from the last convolutional layers of the Regular and Deformable architectures.

### Similarity
- ``NSynth-CKA.ipynb``: Calculate the Linear CKA similarity measure for the various deep features and hand-crafted features.  Plot the results.  CCA $R^2$ and linear regression similarities also calculated.

### Transfer

- ``Analyze-Classifcation.ipynb``: Code to plot and analyze results of the ``Classify-IF.ipynb``, ``Classify-Pitch.ipynb``, ``Classify-Untrained.ipynb`` and ``Classify-Velocity.ipynb`` notebooks.
- ``Classify-IF.ipynb``: Use deep features from all layers trained to classify Instrument Family to classify Instrument Family with logistic regression and to predict Note Pitch and Note Velocity via linear regression.
- ``Classify-Pitch.ipynb``: Use deep features from all layers trained to predict Note Pitch to classify Instrument Family with logistic regression and to predict Note Pitch and Note Velocity via linear regression.
- ``Classify-Untrained.ipynb``: Use deep features from all layers of the untrained Regular and Deformable architectures to classify Instrument Family, and predict Note Pitch and Note Velocity.
- ``Classify-Velocity.ipynb``: Use deep features from all layers trained to predict Note Velocity to classify Instrument Family with logistic regression and to predict Note Pitch and Note Velocity via linear regression.
- ``NSynth-CKA.ipynb``: Linear CKA similarities with deep features from all layers, from the architectures trained to predict Note Pitch and Note Velocity, and comparisons to untrained features and the hand-crafted features.


