# from dataset.dataset_generator import ArtificialDataset
from dataset.generator import generateDataset
from models.model_factory import model_factory
from models.inference import runSeq2SeqInference
import tensorflow as tf
import numpy as np

# Embedding dims to represent a number
embedding_dims = 64
# Output dimensionality of LSTM
lstm_units = 64

# Training and validations size
num_samples_training = 150_000
num_sample_validation = 50_000

# Length of input sequence
sample_length = 10

# Upper bound (range.random()) to generate a number
max_value = 100

vocab_size = max_value + 2 # +2 for SOS and EOS
input_length = sample_length + 1 # For special chars at the beggining of input

def main() -> None:
    print('Generating Dataset')
    # generate training dataset
    trainEncoderInput, trainDecoderInput, trainDecoderOutput = generateDataset(num_samples_training, sample_length, max_value, vocab_size)

    # generate validation dataset
    valEncoderInput, valDecoderInput, valDecoderOutput = generateDataset(num_sample_validation, sample_length, max_value, vocab_size)
    print('Dataset Generated!')

    # Get model name
    try:
        modelName = sys.argv[1]
    except:
        # Use pointer by default
        modelName = 'pointer'

    # Create model
    model = model_factory(modelName, vocab_size, input_length, embedding_dims, lstm_units)
    model.summary()
    
    model.fit(
        x=[trainEncoderInput, trainDecoderInput],
        y=trainDecoderOutput,
        epochs=15,
        batch_size=128,
        shuffle=True,
        validation_data=([valEncoderInput, valDecoderInput], valDecoderOutput),
        # callbacks = [tensorboard_callback]
    )

    # Test the model
    num_samples_tests = 200
    correctPredictions = 0
    wrongPredictions = 0
    trainEncoderInput, _, _ = generateDataset(num_samples_tests, sample_length, max_value, vocab_size)
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
        predictedAnswer = runSeq2SeqInference(model, inputEntry, vocab_size, input_length, max_value)
        print(predictedAnswer)

        # Compute the diff between the correct answer and predicted
        # If diff is equal to 0 then numbers are sorted correctly
        diff = []
        for index, _ in enumerate(correctAnswer):
            diff.append(correctAnswer[index] - predictedAnswer[index])
        
        # If all numbers are equal to 0
        if (all(result== 0 for (result) in diff)):
            correctPredictions += 1
            print('______________________OK__________________________')
        else:
            wrongPredictions += 1
            print('_____________________WRONG!_______________________')
    
    print(f"Correct Predictions: {correctPredictions/num_samples_tests} || Wrong Predictions: {wrongPredictions/num_samples_tests}")

if __name__ == "__main__":
    main()
