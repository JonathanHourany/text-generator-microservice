"""Text generator"""

import json
import tensorflow as tf
from argparse import ArgumentParser
from tools.text_generators import generate_text
from tools.types import TFModel


EMBEDDING_DIMENSIONS = 256 # The embedding dimension
NUM_RNN_UNITS = 1024

with open("feature_maps/char2idx.json") as fp:
    CHAR2EMBEDD_MAP = json.load(fp)

with open("feature_maps/idx2char.json") as fp:
    EMBEDD2CHAR_MAP = json.load(fp)

# Got to compensate for JSON and it's inability to store ints as keys >.>
EMBEDD2CHAR_MAP = {int(k):v for k, v in EMBEDD2CHAR_MAP.items()}
VOCAB_SIZE = len(CHAR2EMBEDD_MAP)


def build_model(vocab_size: int, embedding_dim: int, rnn_units: int, batch_size: int) -> TFModel:
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
        ])

    return model


def load_model_from_weights(model_weights_path: str, vocab_size: int=VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIMENSIONS,
                            rnn_units: int = NUM_RNN_UNITS, batch_size: int = 1) -> TFModel:
    model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=batch_size)
    model.load_weights(tf.train.latest_checkpoint(model_weights_path))
    model.build(tf.TensorShape([1, None]))

    return model


def main(start_string: str, model_weights_path: str):
    model = load_model_from_weights(model_weights_path, VOCAB_SIZE, EMBEDDING_DIMENSIONS, NUM_RNN_UNITS, batch_size=1)

    return generate_text(model, start_string, text2embedd=CHAR2EMBEDD_MAP, embedd2text=EMBEDD2CHAR_MAP)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("start_text", help="Starting text to seed text generation")
    parser.add_argument("--model_weights_path", default='model_weights/',
                        help="Path to model weights. Default checks model_weights for latest checkpoint")

    args = parser.parse_args()

    if args.model_weights_path == 'model_weights/':
        args.model_weights_path = tf.train.latest_checkpoint(args.model_weights_path)
    print(args.model_weights_path)
    print(main(args.start_text, args.model_weights_path))
