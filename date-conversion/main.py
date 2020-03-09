# import dataset.generator
from dataset.date_format import INPUT_VOCAB, OUTPUT_VOCAB, INPUT_LENGTH, OUTPUT_LENGTH, INPUT_FNS, dateTupleToYYYYDashMMDashDD
from dataset import generator
from models.inference import runSeq2SeqInference
from models.model_factory import model_factory

from datetime import datetime
from tensorflow import keras
import sys

minYear = '1950-01-01'
maxYear = '2050-01-01'

embeddingDims = 64
lstmUnits = 64

def main(minYear: str, maxYear: str) -> None:
    # Generate dataset
    trainEncoderInput, trainDecoderInput, trainDecoderOutput, valEncoderInput, valDecoderInput, valDecoderOutput, testDateTuples = generator.generateDataSet(
        minYear, maxYear)

    # Get model name
    try:
        modelName = sys.argv[1]
    except:
        # Use Luong by default
        modelName = 'luong'
    
    # Create Model
    model = model_factory(modelName, len(INPUT_VOCAB), len(
        OUTPUT_VOCAB), INPUT_LENGTH, OUTPUT_LENGTH, embeddingDims, lstmUnits)
    
    # Show model stats
    model.summary(line_length=200)

    # Tensorboard callbacks
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(
        x=[trainEncoderInput, trainDecoderInput],
        y=trainDecoderOutput,
        epochs=3,
        batch_size=128,
        shuffle=True,
        validation_data=([valEncoderInput, valDecoderInput], valDecoderOutput),
        # callbacks = [tensorboard_callback]
    )

    # Test the model
    numTests = 10
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
            outputStr = runSeq2SeqInference(model, inputStr)

            print(f"Predicted Answer: {outputStr}")
            if (outputStr == correctAnswer):
                print('CORRECT')
            else:
                print('WRONG!')


if __name__ == "__main__":
    main(minYear, maxYear)
