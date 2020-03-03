# Sequence 2 Sequence with Attention Mechanisms
This repo contains implementation of:
- Classical Sequence 2 Sequence model without attention
- Luong's Dot Attention
- Bahdanau's Attention

## Date Conversion Problem
Convert dates in different formats (e.g., `08/30/21`, `080120`, `AUG 01, 2020`) into ISO standard (e.g., `2021-08-30`, `2020-08-01`) format.

### Problem Stats
- Input vocabulary size: 35
- Input length: 12
- Output vocabulary size: 13
- Output length: 10

### Running 
```bash
python date-conversion/main.py <model-name> # One of "seq2seq", "luong" or "bahdanau". If not provided "luong" will be used
```

### Run Unit Tests
```bash
python date-conversion/tests/runner.py
```

## Sorting Numbers

## Useful Links
- Tensorflow.js [data-conversion-attention](https://github.com/tensorflow/tfjs-examples/tree/master/date-conversion-attention) example. I've simply ported the dataset generation script and Luong's attention to Python. All the credit goes to the TF team and the people that built the model.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [Pointer Networks](https://arxiv.org/abs/1506.03134)
- [Neural machine translation with attention](https://www.tensorflow.org/tutorials/text/nmt_with_attention)
- [Attention Mechanism](https://blog.floydhub.com/attention-mechanism/)
- [Attn: Illustrated Attention](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)


## Setting the environment and installing the dependencies
Follow Tensorflow's [installation guide](https://www.tensorflow.org/install/pip) to set the environment and get things ready.

> I'm using Python v3.6 and Tensorflow v2.1

## Pytorch Implementation
Please check [fmstam](https://github.com/fmstam)'s [repo](https://github.com/fmstam/seq2seq_with_deep_attention)

