# from dataset.dataset_generator import ArtificialDataset
from dataset.generator import generateDataset
from models.model_factory import model_factory
from models.inference import runSeq2SeqInference
from utils.read_configs import get_configs

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
num_samples_training = 50_000
num_sample_validation = 5000

# Length of input sequence
sample_length = 10

# Upper bound (range.random()) to generate a number
max_value = 100

vocab_size = max_value + 3  # +3 for MASK, SOS and EOS
input_length = sample_length + 1  # For special chars at the beggining of input

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

    # Test the model
    num_samples_tests = 200
    correctPredictions = 0
    wrongPredictions = 0
    trainEncoderInput, _, _  = generateDataset(
        configs['num_samples_tests'],
        configs['sample_length'],
        configs['min_value'],
        configs['max_value'],
        configs['SOS_CODE'],
        configs['EOS_CODE'],
        configs['vocab_size'])
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
            model,
            inputEntry,
            vocab_size,
            input_length,
            max_value, 
            configs['SOS_CODE'], 
            configs['EOS_CODE'])
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
