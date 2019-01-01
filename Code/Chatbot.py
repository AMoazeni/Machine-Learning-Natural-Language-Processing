# Building a ChatBot with Deep NLP



# Importing libraries
import numpy as np
import tensorflow as tf
import re
import time



# PART 1 - DATA PREPROCESSING


# Importing the dataset and split by lines of dataset
lines = open('../Data/movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('../Data/movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')


# Creating a dictionary that maps each line and its id
# '_line' is a temporary variable only used in the loop
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    
    # Only accept lines with 5 elements to avoid shifting issues
    if len(_line) == 5:
        # Create a key from the first element (line ID) and value from last element (words)
        id2line[_line[0]] = _line[4]


# Creating a list of all of the conversations
conversations_ids = []

# Ignore last row which is empty
for conversation in conversations[:-1]:
    
    # [-1] takes the last element, [1:-1] removes the first and last element (square brakets)
    # Remove single quotes and remove spaces
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    # Split result array by commas, and append to conversation_ids
    conversations_ids.append(_conversation.split(','))


# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"c'mon", "come on", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"'em", "them", text)
    text = re.sub(r"'in", "ing", text)
    text = re.sub(r"'cause", "because", text)
    text = re.sub(r"c'mere", "come here", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))


# Creating a dictionary that maps each word to its number of occurrences
word2count = {}
for question in clean_questions:
    for word in question.split():
        
        # Check if first apperance of word
        if word not in word2count:
            word2count[word] = 1
        
        # Increment number of occurance
        else:
            word2count[word] += 1
            
# Count words occurance in answers           
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1



# Creating two dictionaries that map the questions words and the answers words to a unique integer (tokenization)
# 5% threshold of 20,000 words is 20
threshold_questions = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions:
        questionswords2int[word] = word_number
        word_number += 1


threshold_answers = 20
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answerswords2int[word] = word_number
        word_number += 1



# Adding the last tokens to these two dictionaries
# <PAD> Empty space filling
# <EOS> End of string
# <OUT> Word replacement for uncommon words filtered out by the threshold
# <SOS> Start of string

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1


# Creating the inverse dictionary of the answerswords2int dictionary, used for decoding
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}


# Adding the End Of String token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'


# Translating all the questions and the answers into integers
# and Replacing all the words that were filtered out by <OUT> 
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)


# Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []

# Easier to train shorter questions first, start with 25 word long questions
for length in range(1, 25 + 1):
    # 'enumerate' get a couple containing (index, question)
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            





# PART 2 - BUILDING THE SEQ2SEQ MODEL


# Creating placeholders for the inputs and the targets (answers)
def model_inputs():
    
    # placeolder('type of data', 'dimension of input', 'name of placeholder')
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    
    # 'lr' - Learning Rate
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    
    # 'keep_prob' - Dropout Rate (avoids overfitting)
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    return inputs, targets, lr, keep_prob



# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    
    # Create <SOS> token column for the beginning of target
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    
    # Create Last Column which is all the answers, exclude last column token whichis not needed for decoding
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    
    # '1' means horizontal concatenation
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    
    return preprocessed_targets



# Creating the Encoder RNN

# 'rnn_inputs' - Model inputs which inludes learning_rate, keep_prop etc
# 'rnn_size' - Number of input tensors in the layers
# 'keep_prop' - Used for Dropout Regularization
# 'sequence_length' - List of length of questions in each bach
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    
    # Create a simple LSTM class
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    # Dropout wrapper class for LSTM, typically 20% of neurons are deactivated to mitigate overfitting
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    
    # Multi RNN layer setup
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    
    # Encoder State - Creates a dynamic version of a bi-directional RNN
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state



# Decoding the training set
    
# 'encoder_state' - Recieve this class as an input to decode
# 'decoder_cell' - Cel in the RNN of the decoder
# 'decoder_embedded_input' - Inputs that have embedding enabled (mapping from objects like words to a vector of unique real numbers)
# 'decoding_scope' - It's a data structure that wraps your data called 'variable_scope'
# 'output_function' - Function used to return outputs at the end
# 'keep_prob' - Dropout regularization
# 'batch_size' - Working with batches
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    
    # Initialize 3D matrix containing zeros
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    
    # Get all the Attention Function attributes from the seq2seq library
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    
    # Attention Function decoder
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    
    # Get the output and final state of the decoder
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    
    return output_function(decoder_output_dropout)



# Decoding the test/validation set
    
# Same as above 'decode_training_set' function except use 'attention_decoder_fn_inference' to logically infer context
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions



# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        
        # Initialize weights with a standard deviation = 0.1
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        
        # Set up biases
        biases = tf.zeros_initializer()
        
        # Fully connected output function
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        
        # Get training predictions
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        
        # Test training predictions
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    
    return training_predictions, test_predictions



# Building the seq2seq model

# inputs - Question asked from the chatbot
# targets - Answer to the question
# sequence_length - How long the sequence should be
# answers_num_words - Number of words in answers
# questions_num_words - Number of words in questions
# encoder_embedding_size - Number of dimensions in encoder
# decoder_embedding_size - Number of dimensions in decoder
# questionswords2int - Dictionary of words

def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    
    # Define the Encoder RNN
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    
    # Words2Int preprocessed targets
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    
    # Dimensions of the Embedding Matrix - initialize with random uniform distribution
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    
    # Get decoder embedded input
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    # Get predictions
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
    








# PART 3 - TRAINING THE SEQ2SEQ MODEL



# Setting the Hyperparameters

# Itterations of the entire training process (100+ for good results, no lower than 50)
epochs = 1

# Batches of questions used in training (128 for faster runtime)
batch_size = 64

# Size of Recurrent Neural Network
rnn_size = 512

# Number of Encoder and Decoder layers
num_layers = 3

# Number of columns in the Embedding Matrix
encoding_embedding_size = 512
decoding_embedding_size = 512

# Learning Rate too high means model trained too fast and can't speak properly, too low will be really slow
learning_rate = 0.01

# Percentage by which learning rate is decayed over iterations, '1' means no decay
learning_rate_decay = 0.9

# Cap the the minimum learning rate
min_learning_rate = 0.0001

# Dropout reguralization, probability of neurons to be ignored during training (use 20% for input units, 50% for hidden layers)
keep_probability = 0.5




# Initialize a TensorFlow session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length TO maximum sequence length (25 word sequence)
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)


# Get the training and test predictions reshaped into the correct size
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)


# Setting up the Loss Error, the Optimizer and Gradient Clipping (Cap gradients to prevent expanging/disappearing gradient)
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


# Padding the sequences with the <PAD> token - makes sure the length of the question sequence is the same as the answer sequence
# Question - [ 'Who', 'are',  'you', <PAD>, <PAD>, <PAD>, <PAD>]
# Answer -   [<SOS> 'I',  'am',  'a',  'bot',  '.', <EOS>, <PAD>]
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch


# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]


# Begin Training

# Check training loss every 100 batches
batch_index_check_training_loss = 100

# Check validation loss every half number of batches
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1

# Compute sum of losses every 100 batches
total_training_loss_error = 0

# List of validation loss errors - check all losses, pick minimum
list_validation_loss_error = []

# Each time loss is not improved, stop training. Make it last all the way through epochs by choosing a number higher than epochs (100)
early_stopping_check = 0
early_stopping_stop = 1000

# Save weights
# For Windows users, replace this line of code by: checkpoint = "./chatbot_weights.ckpt"
checkpoint = "chatbot_weights.ckpt"

# Run the TensorFlow session
session.run(tf.global_variables_initializer())

# Begin Training loop for all epochs
for epoch in range(1, epochs + 1):
    
    # Algorithm for one epoch
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        
        # Measure run time
        starting_time = time.time()
        
        # Measure trining loss error
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        
        # Add patch error to total error
        total_training_loss_error += batch_training_loss_error
        
        # Measure batch time
        ending_time = time.time()
        batch_time = ending_time - starting_time
        
        # Average of training loss error of 100 batches
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        
        
        # Average validation loss error at halfway and end of epoch
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    
    
    # Check for early stopping
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break


print("Finished Training!!")




# PART 4 - TESTING THE SEQ2SEQ MODEL
 
 
 
# Loading the weights and Running the session
checkpoint = "../Jupyter Notebook/chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)
 
# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# Setting up the chat
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = ' I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)