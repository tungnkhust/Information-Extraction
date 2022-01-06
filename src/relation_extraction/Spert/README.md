# SpERT: Span-based Entity and Relation Transformer
PyTorch code for SpERT: "Span-based Entity and Relation Transformer". For a description of the model and experiments, see our paper: https://arxiv.org/abs/1909.07755 (published at ECAI 2020).

![alt text](http://deepca.cs.hs-rm.de/img/deepca/spert.png)

## Setup
### Requirements
- Required
  - Python 3.5+
  - PyTorch (tested with version 1.4.0)
  - transformers (+sentencepiece, e.g. with 'pip install transformers[sentencepiece]', tested with version 4.1.1)
  - scikit-learn (tested with version 0.24.0)
  - tqdm (tested with version 4.55.1)
  - numpy (tested with version 1.17.4)
- Optional
  - jinja2 (tested with version 2.10.3) - if installed, used to export relation extraction examples
  - tensorboardX (tested with version 1.6) - if installed, used to save training process to tensorboard
  - spacy (tested with version 3.0.1) - if installed, used to tokenize sentences for prediction

### Fetch data
Fetch converted (to specific JSON format) CoNLL04 \[1\] (we use the same split as \[4\]), SciERC \[2\] and ADE \[3\] datasets (see referenced papers for the original datasets):
```
bash ./scripts/fetch_datasets.sh
```

Fetch model checkpoints (best out of 5 runs for each dataset):
```
bash ./scripts/fetch_models.sh
```
The attached ADE model was trained on split "1" ("ade_split_1_train.json" / "ade_split_1_test.json") under "data/datasets/ade".

## Examples
(1) Train CoNLL04 on train dataset, evaluate on dev dataset:
```
python ./spert.py train --config configs/example_train.conf
```

(2) Evaluate the CoNLL04 model on test dataset:
```
python ./spert.py eval --config configs/example_eval.conf
```

(3) Use the CoNLL04 model for prediction. See the file 'data/datasets/conll04/conll04_prediction_example.json' for supported data formats. You have three options to specify the input sentences, choose the one that suits your needs. If the dataset contains raw sentences, 'spacy' must be installed for tokenization. Download a spacy model via 'python -m spacy download model_label' and set it as spacy_model in the configuration file (see 'configs/example_predict.conf'). 
```
python ./spert.py predict --config configs/example_predict.conf
```

## References
```
[1] Spert, https://github.com/lavis-nlp/spert
```
