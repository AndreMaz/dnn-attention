# from dataset.dataset_generator import ArtificialDataset
from dataset.generator import generateDataset
from models.eager_pointer_with_teacher.model import EagerModel
from models.eager_inference import runSeq2SeqInference
import tensorflow as tf
import numpy as np
import sys

# For plotting
import matplotlib.pyplot as plt

num_epochs = 10
batch_size = 128

# Embedding dims to represent a number
embedding_dims = 64
# Output dimensionality of LSTM
lstm_units = 64

# Training and validations size
num_samples_training = 49920
# num_samples_training = 256


# Length of input sequence
sample_length = 10

# Upper bound (range.random()) to generate a number
max_value = 100

vocab_size = max_value + 3  # +3 for MASK, SOS and EOS
input_length = sample_length + 1  # For special chars at the beggining of input


def main(plotAttention=False) -> None:
    print('Generating Dataset')
    # generate training dataset
    trainEncoderInput, trainDecoderInput, trainDecoderOutput, _ = generateDataset(
        num_samples_training, sample_length, max_value, vocab_size)

    # generate validation dataset
    # valEncoderInput, valDecoderInput, valDecoderOutput = generateDataset(
    #    num_sample_validation, sample_length, max_value, vocab_size)
    print('Dataset Generated!')

    loss_fn = tf.losses.CategoricalCrossentropy()
    optimizer = tf.optimizers.Adam()

    model = EagerModel(vocab_size, embedding_dims, lstm_units)

    losses = []

    # tf.expand_dims(trainEncoderInput[0], axis=0)
    # res = model(tf.expand_dims(trainEncoderInput[0], axis=0), tf.expand_dims(trainDecoderInput[0], axis=0))
    num_batches = int(num_samples_training / batch_size)
    for epoch in range(num_epochs):
        loss_per_epoch = []
        for i in range(num_batches - 1):
            enc_in_batch = trainEncoderInput[i * batch_size: (i+1) * batch_size]
            dec_in_batch = trainDecoderInput[i * batch_size: (i+1) * batch_size]
            dec_out_batch = trainDecoderOutput[i * batch_size: (i+1) * batch_size]

            with tf.GradientTape() as tape:
                predicted = model(enc_in_batch, dec_in_batch)
                loss = loss_fn(dec_out_batch, predicted)
                # Store the loss
                loss_per_epoch.append(loss)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        epoch_loss = np.asarray(loss_per_epoch).mean()
        print(f"Epoch: {epoch} avg. loss: {epoch_loss}")
        losses.append(epoch_loss)

    print(losses)

    # Test the model
    num_samples_tests = 200
    correctPredictions = 0
    wrongPredictions = 0
    trainEncoderInput, _, _, _ = generateDataset(
        num_samples_tests, sample_length, max_value, vocab_size)
    for _, inputEntry in enumerate(trainEncoderInput):
        print('__________________________________________________')

        # print number sequence without EOS
        print(list(inputEntry.numpy().astype("int16")[1:]))

        # Generate correct answer
        correctAnswer = list(inputEntry.numpy().astype("int16"))
        # Remove EOS, sort the numbers and print the correct result
        correctAnswer = correctAnswer[1:]
        correctAnswer.sort()
        print(correctAnswer)

        # Add the batch dimension [batch=1, features]
        inputEntry = tf.expand_dims(inputEntry, 0)

        # Run the inference and generate predicted output
        predictedAnswer, attention_weights = runSeq2SeqInference(
            model, inputEntry, vocab_size, input_length, max_value)
        if (plotAttention == True):
            plotAttention(attention_weights, inputEntry)
        print(predictedAnswer)

        # Compute the diff between the correct answer and predicted
        # If diff is equal to 0 then numbers are sorted correctly
        diff = []
        for index, _ in enumerate(correctAnswer):
            diff.append(correctAnswer[index] - predictedAnswer[index])

        # If all numbers are equal to 0
        if (all(result == 0 for (result) in diff)):
            correctPredictions += 1
            print('______________________OK__________________________')
        else:
            wrongPredictions += 1
            print('_____________________WRONG!_______________________')

    print(
        f"Correct Predictions: {correctPredictions/num_samples_tests} || Wrong Predictions: {wrongPredictions/num_samples_tests}")


def plotAttention(attention_weights, inputEntry):
    # print(attention_weights[0].shape)
    plt.matshow(attention_weights[0])

    xTicksNames = list(inputEntry[0].numpy().astype("int16"))
    inputLength = len(xTicksNames)

    yTicksNames = []
    for i in range(inputLength):
        yTicksNames.append(f"step {i}")

    plt.yticks(range(inputLength), yTicksNames)

    plt.xticks(range(inputLength), xTicksNames)

    plt.ylabel('Pointer Probability')
    plt.xlabel('Input Sequence')

    plt.show(block=True)

if __name__ == "__main__":
    main()
