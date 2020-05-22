# import dataset.generator
from dataset.date_format import INPUT_VOCAB, OUTPUT_VOCAB, INPUT_LENGTH, OUTPUT_LENGTH, INPUT_FNS, dateTupleToYYYYDashMMDashDD
from dataset.generator import generateDataSet
from models.inference import runSeq2SeqInference
from models.model_factory import model_factory
from tensorflow.keras.models import Model
from utils.read_configs import get_configs

from datetime import datetime
from tensorflow import keras
import sys

# For plotting
import matplotlib.pyplot as plt

def main() -> None:
    # Get configs
    configs = get_configs(sys.argv)
    model_name = configs['model_name']

    # Generate dataset
    trainEncoderInput, \
    trainDecoderInput, \
    trainDecoderOutput, \
    valEncoderInput, \
    valDecoderInput, \
    valDecoderOutput,\
    testDateTuples = generateDataSet(configs['min_year'], configs['max_year'])

    # Create Model
    model = model_factory(
        model_name,
        len(INPUT_VOCAB),
        len(OUTPUT_VOCAB),
        INPUT_LENGTH,
        OUTPUT_LENGTH,
        configs['embedding_dims'],
        configs['lstm_units']
    )
    
    # Show model stats
    model.summary(line_length=180)

    # Tensorboard callbacks
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # Run training and validation
    model.fit(
        x=[trainEncoderInput, trainDecoderInput],
        y=trainDecoderOutput,
        epochs=3,
        batch_size=128,
        shuffle=True,
        validation_data=([valEncoderInput, valDecoderInput], valDecoderOutput),
        # callbacks = [tensorboard_callback]
    )

    #### TEST THE MODEL ###

    # Create new model that will also return the attention weights
    # New model will produce two outputs: the actual prediction and the attention weights
    if (model_name != 'seq2seq'): # Sequence 2 Sequence model doesn't have attention so we don't need to do this
        model = Model(
            inputs = model.input,
            # Add attention_weights to the output list
            outputs = [model.output, model.get_layer('decoder').output[1]]
        )

    numTests = 10
    totalTests = numTests*len(INPUT_FNS)
    correctPredictions = 0
    wrongPredictions = 0
    for n in range(numTests):
        for _, fn in enumerate(INPUT_FNS):
            # Generate input string
            inputStr = fn(testDateTuples[n])
            print('\n--------------------------------')
            print(f"Input String: {inputStr}")
            # Generate output date (in ISO format)
            correctAnswer = dateTupleToYYYYDashMMDashDD(testDateTuples[n])
            
            print(f"Correct Answer: {correctAnswer}")
            # Run the inference
            if (model_name != 'seq2seq'):
                outputStr, attention_weights = runSeq2SeqInference(model_name, model, inputStr)

                if (configs['to_plot_attention'] == True):
                    plotAttention(attention_weights, inputStr, outputStr, INPUT_LENGTH, OUTPUT_LENGTH)
            else:
                outputStr = runSeq2SeqInference(model_name, model, inputStr)

            print(f"Predicted Answer: {outputStr}")
            if (outputStr == correctAnswer):
                correctPredictions += 1
                print('CORRECT')
            else:
                wrongPredictions += 1
                print('WRONG!')
    
    print(f"Correct Predictions: {correctPredictions/totalTests} || Wrong Predictions: {wrongPredictions/totalTests}")


def plotAttention(attention_weights, inputStr, outputStr, INPUT_LENGTH, OUTPUT_LENGTH):
    plt.matshow(attention_weights[0])
    
    inputChars = list(inputStr)
    inputLength = len(inputChars)
    diffInput = INPUT_LENGTH - inputLength
    xTicksNames = inputChars + [' '] * diffInput

    outputChars = list(outputStr)
    outputLength = len(outputChars)
    diffOutput = OUTPUT_LENGTH - outputLength
    yTicksNames = outputChars + [' '] * diffOutput

    plt.yticks(range(OUTPUT_LENGTH), yTicksNames)
    
    plt.xticks(range(INPUT_LENGTH), xTicksNames)

    plt.ylabel('Generated word')
    plt.xlabel('Input word')

    plt.show(block=True)

if __name__ == "__main__":
    main()
