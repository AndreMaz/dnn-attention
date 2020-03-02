# import dataset.generator
from dataset.date_format import INPUT_VOCAB, OUTPUT_VOCAB, INPUT_LENGTH, OUTPUT_LENGTH, INPUT_FNS, dateTupleToYYYYDashMMDashDD
from dataset import generator
from models.inference import runSeq2SeqInference
# from models.seq2seq.model import createModel
# from models.luong.model import createModel
from models.bahdanau.model import createModel

from datetime import datetime
from tensorflow import keras

minYear = '1950-01-01'
maxYear = '2050-01-01'

def main(minYear: str, maxYear: str) -> None:
    trainEncoderInput, trainDecoderInput, trainDecoderOutput, valEncoderInput, valDecoderInput, valDecoderOutput, testDateTuples = generator.generateDataSet(
        minYear, maxYear)
    # print(trainEncoderInput.shape)
    # print(trainDecoderInput.shape)
    # print(trainDecoderOutput.shape)

    model = createModel(len(INPUT_VOCAB), len(OUTPUT_VOCAB), INPUT_LENGTH, OUTPUT_LENGTH)

    model.summary()

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    model.fit(
        x = [trainEncoderInput, trainDecoderInput],
        y = trainDecoderOutput,
        epochs = 3,
        batch_size = 128,
        shuffle = True,
        validation_data = ([valEncoderInput, valDecoderInput], valDecoderOutput ),
        # callbacks = [tensorboard_callback]
    )

    numTests = 10
    for n in range(numTests):
        for _, fn in enumerate(INPUT_FNS):
            inputStr = fn(testDateTuples[n])
            print('\n--------------------------------')
            print(f"Input String: {inputStr}")
            correctAnswer = dateTupleToYYYYDashMMDashDD(testDateTuples[n])

            print(f"Correct Answer: {correctAnswer}")
            outputStr = runSeq2SeqInference(model, inputStr)
            
            print(f"Predicted Answer: {outputStr}")
            if (outputStr == correctAnswer):
                print('OK')
            else:
                print('NOT OK')


if __name__ == "__main__":
    main(minYear, maxYear)
