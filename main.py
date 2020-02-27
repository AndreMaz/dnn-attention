# import dataset.generator
from dataset.date_format import INPUT_VOCAB, OUTPUT_VOCAB, INPUT_LENGTH, OUTPUT_LENGTH
from dataset import generator
# from models.seq2seq.model import createModel
from models.luong.model import createModel

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
        epochs = 10,
        batch_size = 128,
        shuffle = True,
        validation_data = ([valEncoderInput, valDecoderInput], valDecoderOutput ),
        # callbacks = [tensorboard_callback]
    )


if __name__ == "__main__":
    main(minYear, maxYear)
