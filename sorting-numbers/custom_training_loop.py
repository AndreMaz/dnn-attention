# from dataset.dataset_generator import ArtificialDataset
from dataset.generator import generateDataset
from models.pointer_network.model import EagerModel
from models.inference import runSeq2SeqInference
from utils.read_configs import get_configs
from utils.tester import tester

import tensorflow as tf
import numpy as np
import sys


def main(plotAttention=False) -> None:
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

    print('Dataset Generated!')

    loss_fn = tf.losses.CategoricalCrossentropy()
    optimizer = tf.optimizers.Adam()

    model = EagerModel(
        configs['vocab_size'],
        configs['embedding_dims'],
        configs['lstm_units']
    )

    losses = []

    print('Training...')
    batch_size = configs['batch_size']
    num_batches = int(configs['num_samples_training'] / batch_size)
    for epoch in range(configs['num_epochs']):
        loss_per_epoch = []
        for i in range(num_batches - 1):
            enc_in_batch = trainEncoderInput[i *
                                             batch_size: (i+1) * batch_size]
            dec_in_batch = trainDecoderInput[i *
                                             batch_size: (i+1) * batch_size]
            dec_out_batch = trainDecoderOutput[i *
                                               batch_size: (i+1) * batch_size]

            with tf.GradientTape() as tape:
                predicted = model(enc_in_batch, dec_in_batch)
                loss = loss_fn(dec_out_batch, predicted)
                # Store the loss
                loss_per_epoch.append(loss)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss = np.asarray(loss_per_epoch).mean()
        print(f"Epoch: {epoch+1} avg. loss: {epoch_loss}")
        losses.append(epoch_loss)

    # print(losses)

    print('Testing...')
    tester(model, configs, eager=True)


if __name__ == "__main__":
    main()
