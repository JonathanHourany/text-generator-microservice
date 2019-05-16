"""Tools used in prediction to generate text from a seed text input"""

import tensorflow as tf
from typing import *

from tools.types import TFModel


def generate_text(model:TFModel, start_string: str, text2embedd: Mapping[str, Any], embedd2text: Mapping[Any, str],
                  num_seq_generate: int = 1000, temperature: float = 1.0) -> str:
    """Generates text from a starting string using a TensorFlow model

    Parameters
    ----------
        model: A fitted Tensorflow model that can accept a single row of input (batch size = 1)
        start_string: Text to start to the generation process
        text2embedd: Mapping to map from text to vector embeddings
        embedd2text: Mapping to map from vector embeddings to text
        num_seq_generate: Number of sequences to generate. If using character embeddings, this would be the amount of
            characters to generate after the start string
        temperature: Amount of novelty in the predicted text. Low temperatures results in more predictable text while
            higher temperatures results in more surprising text

    Return
    -------
        generated_text: Model output
    """
    # Evaluation step (generating text using the learned model)
    num_seq_generate = num_seq_generate

    # Converting our start string to numbers (vectorizing)
    input_eval = [text2embedd[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Experiment to find the best setting.
    temperature = temperature

    # Here batch size == 1
    model.reset_states()
    for i in range(num_seq_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(embedd2text[predicted_id])

    return f"{start_string } {''.join(text_generated)}"
