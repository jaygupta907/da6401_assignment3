import torch
from model import Encoder, Decoder, Seq2Seq
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import CharVocab

input_vocab = CharVocab.load("saved/input_vocab.pt")
output_vocab = CharVocab.load("saved/output_vocab.pt")

encoder = Encoder(len(input_vocab), embed_dim=512, hidden_dim=512, num_layers=2, rnn_type='LSTM')
decoder = Decoder(len(output_vocab), embed_dim=512, hidden_dim=512, num_layers=2, rnn_type='LSTM',use_attention=True,encoder_output_dim=1024)
model = Seq2Seq(encoder, decoder, rnn_type='LSTM').to(DEVICE)
model.load_state_dict(torch.load("saved/seq2seq_model_attention.pt", map_location=DEVICE))
model.eval()

def translate_word(word):
    input_tensor = input_vocab.tensor_from_text(word).to(DEVICE)

    with torch.no_grad():
        output,_= model.translate(input_tensor,output_vocab, max_len=30)
        output_word = output_vocab.decode(output)
        
    return output_word

path = test_path = "datasets/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
output_file = "prediction_attention.tsv"
with open(path, 'r', encoding='utf8') as f,open(output_file, 'w', encoding='utf8') as out_f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        word = parts[1]
        devanagari_word = translate_word(word)
        out_f.write(f"{word}\t{devanagari_word}\n")


