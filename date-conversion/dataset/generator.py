import pandas as pd
import numpy as np
import tensorflow as tf
from math import floor

from dataset.date_format import START_CODE, INPUT_FNS, OUTPUT_VOCAB, encodeInputDateStrings, encodeOutputDateStrings, dateTupleToYYYYDashMMDashDD


def generateOrderedDates(minYear: str, maxYear: str) -> list:
    daterange = pd.date_range(minYear, maxYear)
    dates = []
    for single_date in daterange:
        date: list = single_date.strftime("%Y-%m-%d").split('-')

        for index, value in enumerate(date):
            date[index] = int(date[index])

        dates.append(date)

    return dates


def dateTuplesToTensor(dateTuples, dec_output_one_hot = True):
    # Encoder Input
    inputs = []
    for _, fn in enumerate(INPUT_FNS):
        for _, dateTuple in enumerate(dateTuples):
            formatedDate = fn(dateTuple)
            inputs.append(formatedDate)

    encoderInput = encodeInputDateStrings(inputs)

    # Decoder Input
    isoDates = []
    for _, dateTuple in enumerate(dateTuples):
        isoDates.append(dateTupleToYYYYDashMMDashDD(dateTuple))

    decoderInput = encodeOutputDateStrings(isoDates).astype("float32")

    if not dec_output_one_hot:
        decoderOutput = decoderInput
        decoderOutput = np.tile(decoderOutput, (len(INPUT_FNS), 1)).astype("int32")

    # Remove Last column
    decoderInput = decoderInput[..., :-1]
    # Create a single column with start code
    shift = np.full((decoderInput.shape[0], 1), START_CODE, dtype='float32')
    # Concat the tensors
    decoderInput = np.concatenate((shift, decoderInput), axis=1)
    # Tile to match the encoderInput
    decoderInput = np.tile(decoderInput, (len(INPUT_FNS), 1))

    if dec_output_one_hot:
        # Decoder Output
        decoderOutput = tf.one_hot(
            encodeOutputDateStrings(isoDates),
            len(OUTPUT_VOCAB)
        )
        # Tile to match the encoderInput
        decoderOutput = np.tile(decoderOutput, (len(INPUT_FNS), 1, 1)).astype("int32")

    return encoderInput, decoderInput, decoderOutput


def generateDataSet(minYear="1950-01-01", maxYear="2050-01-01", trainSplit=0.25, validationSplit=0.15, dec_output_one_hot = True):
    dateTuples = generateOrderedDates(minYear, maxYear)
    
    np.random.shuffle(dateTuples)

    numTrain = floor(len(dateTuples)*trainSplit)
    numValidation = floor(len(dateTuples)*validationSplit)

    trainEncoderInput, trainDecoderInput, trainDecoderOutput = dateTuplesToTensor(
        dateTuples[0:numTrain], dec_output_one_hot)

    valEncoderInput, valDecoderInput, valDecoderOutput = dateTuplesToTensor(
        dateTuples[numTrain:numTrain+numValidation], dec_output_one_hot)

    testDateTuples = dateTuples[numTrain+numValidation: len(dateTuples)]

    return trainEncoderInput, trainDecoderInput, trainDecoderOutput, valEncoderInput, valDecoderInput, valDecoderOutput, testDateTuples
