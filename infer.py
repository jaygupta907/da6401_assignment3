import torch
from model import Encoder, Decoder, Seq2Seq
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import CharVocab

input_vocab = CharVocab.load("saved/input_vocab.pt")
output_vocab = CharVocab.load("saved/output_vocab.pt")

encoder = Encoder(len(input_vocab), embed_dim=256, hidden_dim=512, num_layers=4, rnn_type='LSTM')
decoder = Decoder(len(output_vocab), embed_dim=256, hidden_dim=512, num_layers=4, rnn_type='LSTM')
model = Seq2Seq(encoder, decoder, rnn_type='LSTM').to(DEVICE)

model.load_state_dict(torch.load("saved/seq2seq_model.pt", map_location=DEVICE))
model.eval()

def translate_word(word):
    input_tensor = input_vocab.tensor_from_text(word).to(DEVICE)

    with torch.no_grad():
        output_word = model.translate(input_tensor,output_vocab, max_len=30)
        
    return output_word

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer.py <latin_word>")
        sys.exit(1)

    latin_word = sys.argv[1]
    devanagari_word = translate_word(latin_word)
    print(f"Latin: {latin_word} → Devanagari: {devanagari_word}")
