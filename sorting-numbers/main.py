# from dataset.dataset_generator import ArtificialDataset
from dataset.generator import generateDataset
from models.model_factory import model_factory
from models.inference import runSeq2SeqInference
from utils.read_configs import get_configs

import tensorflow as tf
import numpy as np
import sys

# For plotting
# import matplotlib.pyplot as plt
from utils.tester import tester

def main(plotAttention = False) -> None:
    # Get the configs
    configs = get_configs(sys.argv)

    print('Generating Dataset')
    # generate training dataset
    trainEncoderInput, trainDecoderInput, trainDecoderOutput = generateDataset(
        configs['num_samples_training'],
        configs['sample_length'],
        configs['min_value'],
        configs['max_value'],
        configs['SOS_CODE'],
        configs['EOS_CODE'],
        configs['vocab_size']
    )

    # generate validation dataset
    valEncoderInput, valDecoderInput, valDecoderOutput = generateDataset(
        configs['num_samples_validation'],
        configs['sample_length'],
        configs['min_value'],
        configs['max_value'],
        configs['SOS_CODE'],
        configs['EOS_CODE'],
        configs['vocab_size']
    )
    print('Dataset Generated!')

    # Create model
    model = model_factory(
        configs['model_name'],
        configs['vocab_size'],
        configs['input_length'],
        configs['embedding_dims'],
        configs['lstm_units'])
    model.summary(line_length=180)

    model.fit(
        x=[trainEncoderInput, trainDecoderInput],
        y=trainDecoderOutput,
        epochs=configs['num_epochs'],
        batch_size=configs['batch_size'],
        shuffle=True,
        validation_data=([valEncoderInput, valDecoderInput], valDecoderOutput),
        # callbacks = [tensorboard_callback]
    )

    tester(model, configs)

if __name__ == "__main__":
    main()
