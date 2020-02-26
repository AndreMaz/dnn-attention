# import dataset.generator
from dataset.date_format import INPUT_VOCAB, OUTPUT_VOCAB, INPUT_LENGTH, OUTPUT_LENGTH
from dataset import generator
from models.seq2seq.seq2seq import createModel

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

    model.fit(
        x = [trainEncoderInput, trainDecoderInput],
        y = trainDecoderOutput,
        epochs = 2,
        batch_size = 128,
        shuffle = True
    )


main(minYear, maxYear)
