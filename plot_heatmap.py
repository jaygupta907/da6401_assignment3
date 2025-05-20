import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torchvision.utils as vutils
from PIL import Image
from io import BytesIO
from torchvision import transforms

from model import Encoder, Decoder, Seq2Seq
from dataset import CharVocab
from config import get_args



# Load args and model
args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_vocab = CharVocab.load("saved/input_vocab.pt")
output_vocab = CharVocab.load("saved/output_vocab.pt")



# Initialize W&B
wandb.init(project=args.wandb_project,
                entity=args.wandb_entity,
                config=args,
                name="Attention Heatmap")


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

# Test words
test_words = ["bhadaka","rasaaynik",'andhavishvas','jeevraj','slider','navketan','gatisheel','bastiyon','tyohaar']

# Helper: Create a heatmap and convert to tensor
def attention_to_tensor(input_chars, output_chars, attention_weights, title):
    plt.rcParams['font.family'] = 'Lohit Devanagari'
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(attention_weights, xticklabels=input_chars, yticklabels=output_chars,
                cbar=False, cmap="YlOrBr", ax=ax, annot=False)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', rotation=0)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    tensor_image = transforms.ToTensor()(image)
    return tensor_image

# Generate attention tensors
heatmaps = []
for word in test_words:
    input_tensor = input_vocab.tensor_from_text(word).to(DEVICE)

    with torch.no_grad():
        output, attention_weights = model.translate(input_tensor, output_vocab, max_len=30)

    decoded_input = [input_vocab.idx2char[idx] for idx in input_tensor.squeeze(0).cpu().numpy()]
    decoded_output = [output_vocab.idx2char[idx] for idx in output]

    # Strip special tokens
    decoded_input = [c for c in decoded_input if c not in ['<sos>', '<eos>']]
    decoded_output = [c for c in decoded_output if c not in ['<sos>', '<eos>']]
    attention_weights = np.array(attention_weights).squeeze(1)[:-1,2:]

    # Convert attention heatmap to tensor
    tensor_img = attention_to_tensor(decoded_input, decoded_output, attention_weights, f"{word} - {''.join(decoded_output)}")
    heatmaps.append(tensor_img)

# Stack and create grid
grid = vutils.make_grid(heatmaps, nrow=3, padding=4, normalize=False)

# Log to wandb
wandb.log({"attention_grid": wandb.Image(grid)})
