import re
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import model_loader


def lowercase_and_html_escape(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def create_model_from_scratch():
    # download tar with dataset and remove unused files, commented out as we already downloaded them:
    # url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    #
    # dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
    #                                   untar=True, cache_dir='.',
    #                                   cache_subdir='')
    #
    # dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    # train_dir = os.path.join(dataset_dir, 'train')
    # remove_dir = os.path.join(train_dir, 'unsup')
    # shutil.rmtree(remove_dir)

    # It is recommended to split datasets into training, validation and test groups, but the tar includes only training
    # and test, so we have to split them manually

    # split training set into training and validation sets
    print('######################################### Splitting into datasets...')
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    print('###################################### Example from training dataset:')
    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(3):
            print("Review: ", text_batch.numpy()[i])
            print("Label: ", label_batch.numpy()[i])

    print("Label 0 corresponds to", raw_train_ds.class_names[0])
    print("Label 1 corresponds to", raw_train_ds.class_names[1])

    print('###################################### Preparing validation and test datasets...')
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    # create test dataset
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)

    # create Text Vectorization layer with custom string standardization (see above)
    print('################################################ Creating Text Vectorization layer and adapting dict...')
    max_features = 10000  # number of max distinct words to be extracted from a dataset
    sequence_length = 250  # size of output sequence, constant regardless of number of tokens extracted from a sample

    vectorization_layer = TextVectorization(
        standardize=lowercase_and_html_escape,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_text_without_labels = raw_train_ds.map(lambda text, label: text)
    vectorization_layer.adapt(train_text_without_labels)  # create a dictionary of distinct words from the test set

    def vectorization_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorization_layer(text), label

    print('############################### Example of a vectorized text:')
    text_batch, label_batch = next(iter(raw_train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review", first_review)
    print("Label", raw_train_ds.class_names[first_label])
    print("Vectorized review", vectorization_text(first_review, first_label))

    print('#################################### Vectorize training, validation and test datasets:')
    train_ds = raw_train_ds.map(vectorization_text)
    val_ds = raw_val_ds.map(vectorization_text)
    test_ds = raw_test_ds.map(vectorization_text)

    print('##################################### Caching for performance...')
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print('#################################### Creating model...')
    embedding_dim = 16
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)])

    model.summary()

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    print('################################### Fitting a model into train (and validation data)')
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    print('###################################### Evaluating a model')
    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    print('###################################### Prepare a model for exporting')
    export_model = tf.keras.Sequential([
        vectorization_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    print('######################################### Predict for a new string:')
    example = 'It was a disaster. I cannot recommend this movie because it was so painful to watch.'
    print(example)
    value = export_model.__call__([example])
    print("Prediction: ", value)

    tf.keras.models.save_model(model, 'savedModel/data', overwrite=True)


if __name__ == '__main__':
    model = model_loader.get_model()
    print('Obtained model')
    print(model.predict(['It was awful. I regret going to the cinema.']))
    print(model.predict(['I think this movie is excellent. One of the best I have ever seen.']))
    print(model.predict(['I am biased. Some parts in this movie were outstanding, while others '
                         'were so terrible I had to hide my disgust']))
