import numpy as np
import sys
from torchtext.vocab import GloVe
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset , DataLoader
# from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

torch.manual_seed(42)

glove_file_path = 'glove.6B.100d.txt'

def load_glove_embeddings(file_path):
    word_embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip the first line

        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word_embeddings[word] = vector
    return word_embeddings

glove_embeddings = load_glove_embeddings(glove_file_path)

#----------------------------------------------------------
#--------------------- Main Execution ---------------------
# ---------------------------------------------------------


if (len(sys.argv) != 2):
    print("Correct Command: pos_tagger.py <-f/-r>")
    sys.exit(1)


model_type = sys.argv[1][1]


if(model_type == 'f'):

    p = 1
    s = 1
    start_tag = '<S>'
    end_tag = '</S>'
    unknown_token = '<UKN>'
    padding_token = '<PAD>'
    BATCH_SIZE = 64
    round_precision = 5
    EPOCHS = 10
    embedding_dim = 100
    HIDDEN_LAYER_DIMENSION = 100

    glove_file_path = 'glove.6B.100d.txt'

    unknown_word_embedding = np.zeros(embedding_dim)
    start_tag_embedding = np.round(np.random.normal(size=(embedding_dim,)) , round_precision)
    end_tag_embedding = np.round(np.random.normal(size=(embedding_dim)) , round_precision)
    
    def load_glove_embeddings(file_path):
        word_embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip the first line

            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                word_embeddings[word] = vector
        return word_embeddings
    
    global_embedding = load_glove_embeddings(glove_file_path)

    class ffnn_model(nn.Module):

        def __init__(self, input_dim , hidden_dim, output_dim):
            super(ffnn_model, self).__init__()

            self.l1 = nn.Linear(input_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, 2*hidden_dim)
            self.l3 = nn.Linear(2*hidden_dim, 3*hidden_dim)
            self.l4 = nn.Linear(3*hidden_dim, 2*hidden_dim)
            self.l5 = nn.Linear(2*hidden_dim, output_dim)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)


        def forward(self, x):
            x = x.to(self.l1.weight.dtype)
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            out = self.relu(out)
            out = self.l3(out)
            out = self.relu(out)
            out = self.l4(out)
            out = self.relu(out)
            out = self.l5(out)

            return out

    loaded_model  = ffnn_model((p+s+1)*embedding_dim , HIDDEN_LAYER_DIMENSION , 13)
    loaded_model.load_state_dict(torch.load('ffnn_model_cpu.pt'))
    # print(loaded_model)

    pos_tag_embedding = {'ADP': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        'PRON': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                        'ADJ': [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        'VERB': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                        'CCONJ': [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        'PART': [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                        'ADV': [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        'NOUN': [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                        'INTJ': [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                        'PROPN': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                        'DET': [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                        'NUM': [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                        'AUX': [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]}

    start_tag_list = []
    end_tag_list = []

    for i in range(p):
        start_tag_list.append(start_tag)

    for i in range(s):
        end_tag_list.append(end_tag)


    try:
        while(1):
            sentence = input("Enter your sentence:   ")
            tokens = sentence.split(' ')
            punc = [',', '.', '!', '?']
            tokens = [word.lower() for word in tokens if word not in punc]
            sentence_length = len(tokens)

            new_sentence = start_tag_list + tokens + end_tag_list
            sentence_embedding = []

            for token in new_sentence :
                if token == start_tag:
                    sentence_embedding.append(start_tag_embedding)

                elif token == end_tag:
                    sentence_embedding.append(end_tag_embedding)

                elif token not in global_embedding:
                    sentence_embedding.append(unknown_word_embedding)

                else:
                    sentence_embedding.append(np.array(global_embedding[token]))

            single_entry = []
            for index in range(p, sentence_length+p):

                p_1_s_token = sentence_embedding[index-p : index+s+1]
                single_entry.append(p_1_s_token)

            answer = []

            for entry in single_entry:

                flattern_array = torch.tensor(np.array(entry).flatten())
                # flattern_array = torch.tensor(np.array(entry).flatten()).long()
                output = loaded_model(flattern_array)
                _, predicted = torch.max(output, 0)

                for tag , embedding in pos_tag_embedding.items():

                    if predicted == torch.argmax(torch.tensor(embedding), dim=0):
                        answer.append(tag)
                        break

            for i in range(sentence_length):
                print(tokens[i] , "   ->    ", answer[i])

    except KeyboardInterrupt:
        print("\n\nLoop Ternimated!!")

elif(model_type == 'r'):

    ##------------------ Global Variable -----------------------------------------------------
    start_tag = '<S>'
    end_tag = '</S>'
    unknown_token = '<UKN>'
    padding_token = '<PAD>'

    BATCH_SIZE = 64
    round_precision = 5
    EPOCHS = 10
    embedding_dim = 200
    hidden_layer_dim = 10
    learning_rate = 1e-3
    bidirectional = True
    num_layer = 4 

    # glove_file_path = '/content/glove.6B.100d.txt'

    unknown_word_embedding = np.zeros(embedding_dim)
    start_tag_embedding = np.round(np.random.normal(size=(embedding_dim,)) , round_precision)
    end_tag_embedding = np.round(np.random.normal(size=(embedding_dim)) , round_precision)

    file = open('en_atis-ud-train.conllu' , 'r')
    train = file.read()
    file.close()

    def preprocssing(data):

        sentences = []   # return structure

        currrent_sentence = []
        sentence_token = []
        token_pos = []

        pos_tags = []
        word_freq = dict()

        lines = data.strip().split('\n')

        for line in lines:

            if(len(line) < 1):
                currrent_sentence.append(sentence_token)
                currrent_sentence.append(token_pos)
                sentences.append(currrent_sentence)

                sentence_token = []
                token_pos = []
                currrent_sentence = []

            elif line.startswith('# sent_id'):
                continue

            elif line.startswith('# text ='):
                continue

            else:
                tokens =line.split('\t')

                if tokens[3] != 'SYM':

                    if tokens[1] in word_freq:
                        word_freq[tokens[1]] += 1
                    
                    else:
                        word_freq[tokens[1]] = 1


                    sentence_token.append(tokens[1])
                    token_pos.append(tokens[3])

                    if tokens[3] not in pos_tags:
                        pos_tags.append(tokens[3])



        currrent_sentence.append(sentence_token)
        currrent_sentence.append(token_pos)
        sentences.append(currrent_sentence)
        return sentences , pos_tags, word_freq


    def get_structure(data, pos_tag, word_freq):

        pos_to_index = dict()
        index_to_pos = dict()
        word_to_index = dict()

        index=0
        
        for pos in pos_tag:
            pos_to_index[pos] = index
            index_to_pos[index] = pos
            index += 1


        word_to_index[start_tag] = 0
        word_to_index[end_tag] = 1
        word_to_index[unknown_token] = 2
        word_to_index[padding_token] = 3

        for word, freq in word_freq.items():
            if freq >= 3 and word not in word_to_index:
                word_to_index[word] = len(word_to_index)

        return pos_to_index, index_to_pos, word_to_index

    train_data, pos, word_freq = preprocssing(train)
    pos_to_index, index_to_pos, word_to_index = get_structure(train_data, pos, word_freq)
    print(len(word_to_index))
    
    class LSTMTagger(torch.nn.Module):
        def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size, num_layers, bidirectional):
            
            super(LSTMTagger, self).__init__()
            
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
            
            if bidirectional:
                self.hidden2tag = nn.Linear(hidden_dim * 2, target_size)
            else:
                self.hidden2tag = nn.Linear(hidden_dim, target_size)
            
        def forward(self, sentence):
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
            tag_scores = F.log_softmax(tag_space, dim=1)
            return tag_scores
        
        def predict(self, sentence):
            with torch.no_grad():
                output = self.forward(sentence)
                return torch.argmax(output, dim=1)
            
    loaded_model = LSTMTagger(embedding_dim, hidden_layer_dim, len(word_to_index), len(pos_to_index), num_layer, bidirectional)
    loaded_model.load_state_dict(torch.load("rnn.pt"))

    sentence = input("Enter your sentence:   ")
    tokens = sentence.split(' ')
    punc = [',', '.', '!', '?']
    tokens = [word.lower() for word in tokens if word not in punc]
    sentence_length = len(tokens)

    answer = []
    test = []

    for word in tokens:
        if word in word_to_index:
            index = word_to_index[word]
        else:
            index = word_to_index[unknown_token]
        test.append(index)

    predicted_label = loaded_model.predict(torch.tensor(test))

    for i in predicted_label:
        # print(int(i))
        answer.append(index_to_pos[int(i)])

    for i in range(sentence_length):
        print(tokens[i] , "   ->    ", answer[i])


# else:
#     print("Invalid Choise!!")
