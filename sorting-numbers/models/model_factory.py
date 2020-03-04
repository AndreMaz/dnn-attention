from models.pointer_network.model import createModel as pointerModel

models = {
    'pointer': pointerModel,
}

def model_factory(name, input_vocab_size, output_vocab_size, input_length, output_length, embedding_dims, lstm_units):
    try:
        createModel = models[name]
        print(f'Using "{name.upper()}" model')
    except:
        raise NameError('Unknown Model Name! Select one of: "seq2seq", "luong" or "bahdanau"')
    
    return createModel(input_vocab_size, output_vocab_size, input_length, output_length, embedding_dims, lstm_units)


if __name__ == "__main__":
    pass