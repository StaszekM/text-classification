import re
import string
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.engine.sequential import Sequential


def lowercase_and_html_escape(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def get_model() -> Sequential:
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

    model = tf.keras.Sequential([
        vectorization_layer,
        importedModel,
        layers.Activation('sigmoid')
    ])

    model.compile(loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

    return model
