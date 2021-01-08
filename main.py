# import the dependencies
import tensorflow as tf
assert tf.__version__.startswith('2')
import tensorflow_datasets as tfds
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import transformer

# get the dataset
path_to_zip = tf.keras.utils.get_file('cornell_movie_dialogs.zip',
                            origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
                            extract=True)

path_to_dataset = os.path.join(os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")
path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')

# MAX_SAMPLES = 100000

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    #creates space between word and punctuation
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    #replace everything that is not a-z, A-Z, or punctuation with a space
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

def load_conversations():
    # creating and storing key value pairs into a dictionary of line id to corresponding text
    id2line = {}
    with open(path_to_movie_lines, errors='ignore') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]
    
    inputs, outputs = [], []
    with open(path_to_movie_conversations, 'r') as file:
        lines = file.readlines()
    parts = lines[0].replace('\n', '').split(' +++$+++ ')
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        # get a conversation from a list of line IDs
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
            # if len(inputs) >= 50000:
            #     return inputs, outputs
    return inputs, outputs

questions, answers = load_conversations()

print('Sample question: {}'.format(questions[17]))
print('Sample answer: {}'.format(answers[17]))

# Build tokenizer using tfds using both questions and answers
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size = 2**13)

# Define start and end tokens that indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2
print('Tokenized sample question: {}'.format(tokenizer.encode(questions[17])))

# Maximum sentence length
MAX_LENGTH = 50

# Tokenize, filter, and pad sentences
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        #tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)
    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter(questions, answers)

print('Vocab size: {}'.format(VOCAB_SIZE))
print('Number of samples: {}'.format(len(questions)))

BATCH_SIZE = 64
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

# Cache the dataset
dataset = dataset.cache()
# Shuffle the dataset
dataset = dataset.shuffle(BUFFER_SIZE)
# Divide the dataset into batches
dataset = dataset.batch(BATCH_SIZE)
# Prefetch the dataset
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print(dataset)

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 16
UNITS = 512
DROPOUT = 0.2

# Create the model
model = transformer.transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

# Define the loss function using Sparse Categorical Cross Entropy
def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)
    # Apply the mask correctly
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

# Create a learning rate schedule based on research paper's equation
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

sample_learning_rate = CustomSchedule(d_model=128)

# plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# plt.show()

# Instantiate a learning rate
learning_rate = CustomSchedule(D_MODEL)
# Use the Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# Define the accuracy function
def accuracy(y_true, y_pred):
    # ensure labels have shape (batch size, max length - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

model.summary()

EPOCHS = 40
model.fit(dataset, epochs=EPOCHS)

def evaluate(sentence):
    # Preprocess the sentence with the same method for the input
    sentence = preprocess_sentence(sentence)
    # Tokenize the sentence and add the tokens
    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
    # Create the output
    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)
        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break
        # concatenated the predicted_id to the output which is given to the decoder as its input
        output = tf.concat([output, predicted_id], axis=-1)
    
    return tf.squeeze(output, axis=0)

def predict(sentence):
    # Get the predicted sentence
    prediction = evaluate(sentence)
    # Decode the prediction into words
    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
    # Print the inputs and outputs and return the sentence
    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

output = predict('Where have you been?')
output = predict("It's a trap")

# feed the model with its previous output
sentence = 'I am not crazy, my mother had me tested.'
for _ in range(5):
    sentence = predict(sentence)
    print('')

model.save_weights('Models/transformer_model_weights')

# 1/2: Load models
# 1/3: Host