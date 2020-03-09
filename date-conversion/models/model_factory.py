from models.seq2seq.model import createModel as seq2seqModel
from models.luong.model import createModel as luongModel
from models.bahdanau.model import createModel as bahdanauModel

models = {
    'seq2seq': seq2seqModel,
    'luong': luongModel,
    "bahdanau": bahdanauModel
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