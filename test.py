import torch
from model import Encoder, Decoder, Seq2Seq
import sys
from config import get_args


args = get_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import CharVocab

input_vocab = CharVocab.load("saved/input_vocab.pt")
output_vocab = CharVocab.load("saved/output_vocab.pt")

encoder = Encoder(len(input_vocab), 
                    embed_dim=args.input_embedding,
                    hidden_dim=args.hidden_layer_size,
                    num_layers=args.encoder_layers, 
                    rnn_type=args.cell_type,
                    bidirectional=args.encoder_bidirectional, 
                    dropout=args.dropout)
decoder = Decoder(len(output_vocab), 
                      embed_dim=args.input_embedding, 
                      hidden_dim=args.hidden_layer_size, 
                      num_layers=args.decoder_layers, 
                      rnn_type=args.cell_type,
                      bidirectional=args.decoder_bidirectional,
                      dropout=args.dropout,
                      use_attention=args.use_attention,
                      encoder_output_dim=args.hidden_layer_size * 2 if args.encoder_bidirectional else args.hidden_layer_size)
model = Seq2Seq(encoder, decoder, rnn_type=args.cell_type).to(DEVICE)
model.load_state_dict(torch.load("saved/seq2seq_model_attention.pt", map_location=DEVICE))
model.eval()

def translate_word(word):
    input_tensor = input_vocab.tensor_from_text(word).to(DEVICE)
    with torch.no_grad():
        output, _ = model.translate(input_tensor, output_vocab, max_len=30,beam_width=3)

        output_word = output_vocab.decode(output)
        
    return output_word

# Paths for test data and output predictions
path = test_path = "datasets/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
output_file = "predictions/prediction_attention.tsv"

# Initialize counters for accuracy
total_chars = 0
correct_chars = 0
total_words = 0
correct_words = 0

# Open test data and output file
with open(path, 'r', encoding='utf8') as f, open(output_file, 'w', encoding='utf8') as out_f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        word = parts[1]  # Latin word
        ground_truth = parts[0]  # Devanagari word
        devanagari_word = translate_word(word)  # Predicted word

        # Write predictions to the output file
        out_f.write(f"{word}\t{devanagari_word}\t{ground_truth}\n")

        # Character-level accuracy
        total_chars += len(ground_truth)
        correct_chars += sum(1 for pred_char, gt_char in zip(devanagari_word, ground_truth) if pred_char == gt_char)

        # Word-level accuracy
        total_words += 1
        if devanagari_word == ground_truth:
            correct_words += 1

# Calculate accuracies
char_accuracy = correct_chars / total_chars * 100
word_accuracy = correct_words / total_words * 100

# Print results
print(f"Character-level Accuracy: {char_accuracy:.2f}%")
print(f"Word-level Accuracy: {word_accuracy:.2f}%")