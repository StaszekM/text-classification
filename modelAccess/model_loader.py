import re
import string
import tensorflow as tf
from typing import Tuple, Callable, Optional
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import numpy as np

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.engine.sequential import Sequential


def lowercase_and_html_escape(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def most_influential_words_factory(weights, vocabulary, dense_layer_weights):
    most_influential_dense_weight_index = np.argmax(np.ndarray.flatten(dense_layer_weights))

    def get_most_influential_words(text: str, is_positive: bool) -> Optional[list]:
        inputText = list(dict.fromkeys(bytes.decode(lowercase_and_html_escape(text).numpy()).split(' ')))

        def filter_from_list(word):
            try:
                vocabulary.index(word)
            except ValueError:
                return False
            return True

        inputText = list(filter(filter_from_list, inputText))

        weightedText = [weights[vocabulary.index(word)][most_influential_dense_weight_index] for word in inputText]
        wordsWithWeights = list(zip(inputText, weightedText))
        wordsWithWeights.sort(key=lambda pair: pair[1])

        top_count = 5
        if len(wordsWithWeights) < top_count:
            return None
        if is_positive:
            return list(map(lambda pair: pair[0], wordsWithWeights[-top_count:]))[::-1]
        return list(map(lambda pair: pair[0], wordsWithWeights[:top_count]))

    return get_most_influential_words


def get_model() -> Tuple[Sequential, Callable[[str, bool], Optional[list]]]:
    importedModel = tf.keras.models.load_model('savedModel/data')

    max_features = 10000  # number of max distinct words to be extracted from a dataset
    sequence_length = 250  # size of output sequence, constant regardless of number of tokens extracted from a sample

    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    vectorization_layer = TextVectorization(
        standardize=lowercase_and_html_escape,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_text_without_labels = raw_train_ds.map(lambda text, label: text)
    vectorization_layer.adapt(train_text_without_labels)  # create a dictionary of distinct words from the test set

    weights = importedModel.layers[0].get_weights()[0]
    vocabulary = vectorization_layer.get_vocabulary()
    dense_layer_weights = importedModel.layers[4].get_weights()[0]
    get_most_influential_words = most_influential_words_factory(weights, vocabulary, dense_layer_weights)

    model = tf.keras.Sequential([
        vectorization_layer,
        importedModel,
        layers.Activation('sigmoid')
    ])

    model.compile(loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

    return model, get_most_influential_words
