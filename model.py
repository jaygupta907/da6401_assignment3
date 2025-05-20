import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming Attention, Encoder, and Decoder classes remain unchanged
class Attention(nn.Module):
    def __init__(self, decoder_hidden_dim, encoder_output_dim):
        super(Attention, self).__init__()
        self.attn_W1 = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.attn_W2 = nn.Linear(encoder_output_dim, decoder_hidden_dim)
        self.attn_v = nn.Linear(decoder_hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        enc_proj = self.attn_W2(encoder_outputs)
        dec_proj = self.attn_W1(decoder_hidden).unsqueeze(1)
        attn_input = torch.tanh(dec_proj + enc_proj)
        scores = self.attn_v(attn_input).squeeze(2)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_dim, hidden_dim, num_layers=1, rnn_type='LSTM', bidirectional=True, dropout=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.rnn_type = rnn_type
        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, input_seq):
        embedded = self.dropout(self.embedding(input_seq))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_dim, hidden_dim, encoder_output_dim, use_attention=False, num_layers=1, rnn_type='LSTM', bidirectional=False, dropout=0.2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.rnn_type = rnn_type
        self.use_attention = use_attention
        self.encoder_output_dim = encoder_output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if use_attention:
            self.attention = Attention(hidden_dim, encoder_output_dim)
            rnn_input_size = embed_dim + encoder_output_dim
        else:
            self.attention = None
            rnn_input_size = embed_dim

        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(rnn_input_size, hidden_dim, num_layers,
                           batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input_token)).squeeze(1)
        if self.use_attention:
            if self.rnn_type == 'LSTM':
                h_att = hidden[0][-1]
            else:
                h_att = hidden[-1]
            context, _ = self.attention(h_att, encoder_outputs)
            rnn_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
        else:
            rnn_input = embedded.unsqueeze(1)

        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, rnn_type='LSTM'):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_type = rnn_type

        # Linear layer to project encoder hidden state to decoder hidden state
        encoder_hidden_size = encoder.hidden_dim * (2 if encoder.bidirectional else 1)
        decoder_hidden_size = decoder.hidden_dim * (2 if decoder.bidirectional else 1)
        self.hidden_projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)

    def _init_decoder_hidden(self, encoder_hidden):
        """Initialize decoder hidden state from encoder hidden state."""
        if self.rnn_type == 'LSTM':
            h, c = encoder_hidden
            h = self._project_hidden(h)
            c = self._project_hidden(c)
            return h, c
        else:
            return self._project_hidden(encoder_hidden)

    def _project_hidden(self, hidden):
        """Project and adjust encoder hidden state to match decoder's expected shape."""
        # hidden: (num_layers_encoder * num_directions, batch_size, hidden_dim)
        batch_size = hidden.size(1)
        encoder_num_layers = self.encoder.num_layers
        decoder_num_layers = self.decoder.num_layers
        decoder_hidden_size = self.decoder.hidden_dim * (2 if self.decoder.bidirectional else 1)

        # Reshape to separate layers and directions
        if self.encoder.bidirectional:
            # (num_layers * 2, batch_size, hidden_dim) -> (num_layers, 2, batch_size, hidden_dim)
            hidden = hidden.view(encoder_num_layers, 2, batch_size, -1)
            # Take the last layer: (2, batch_size, hidden_dim)
            hidden = hidden[-1]
            # Concatenate forward and backward: (batch_size, 2 * hidden_dim)
            hidden = hidden.permute(1, 0, 2).reshape(batch_size, -1)
        else:
            # (num_layers, batch_size, hidden_dim) -> (num_layers, batch_size, hidden_dim)
            hidden = hidden.view(encoder_num_layers, batch_size, -1)
            # Take the last layer: (batch_size, hidden_dim)
            hidden = hidden[-1]

        # Project to decoder hidden size
        hidden = self.hidden_projection(hidden)  # (batch_size, decoder_hidden_size)

        # Repeat to match decoder's number of layers
        hidden = hidden.unsqueeze(0)  # (1, batch_size, decoder_hidden_size)
        if decoder_num_layers > 1:
            hidden = hidden.repeat(decoder_num_layers, 1, 1)
        return hidden.contiguous()

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        output_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, output_vocab_size, device=trg.device)

        encoder_outputs, hidden = self.encoder(src)
        hidden = self._init_decoder_hidden(hidden)

        input_token = trg[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    def translate(self, input_tensor, output_vocab, max_len=30, beam_width=1):
        self.eval()
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            device = input_tensor.device

            initial_hidden = self._init_decoder_hidden(encoder_hidden)
            beam = [([output_vocab.char2idx['<sos>']], 0.0, initial_hidden)]
            completed = []
            attention_weights = []

            for _ in range(max_len):
                all_candidates = []
                for seq, score, hidden in beam:
                    if seq[-1] == output_vocab.char2idx['<eos>']:
                        completed.append((seq, score / ((len(seq) + 5) / 6) ** 0.7))
                        continue

                    trg_tensor = torch.tensor([seq[-1]], device=device).unsqueeze(0)
                    output, new_hidden = self.decoder(trg_tensor, hidden, encoder_outputs)

                    if self.decoder.use_attention:
                        h_att = new_hidden[0][-1] if self.rnn_type == 'LSTM' else new_hidden[-1]
                        context, attn_weights = self.decoder.attention(h_att, encoder_outputs)
                        attention_weights.append(attn_weights.cpu().numpy())

                    output = F.log_softmax(output, dim=1)
                    topk_scores, topk_indices = output.topk(beam_width)

                    for k in range(beam_width):
                        new_seq = seq + [topk_indices[0, k].item()]
                        new_score = score + topk_scores[0, k].item()
                        normalized_score = new_score / ((len(new_seq) + 5) / 6) ** 0.7
                        all_candidates.append((new_seq, new_score, new_hidden, normalized_score))

                # Select top beam_width candidates
                all_candidates = sorted(all_candidates, key=lambda x: x[3], reverse=True)[:beam_width]
                beam = [(seq, score, hidden) for seq, score, hidden, _ in all_candidates]

                # Early stopping
                if not beam or all(seq[-1] == output_vocab.char2idx['<eos>'] for seq, _, _ in beam):
                    break

            # Include completed sequences in final selection
            all_candidates = [(seq, score) for seq, score in completed]
            all_candidates.extend([(seq, score / ((len(seq) + 5) / 6) ** 0.7) for seq, score, _ in beam])
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            
            best_seq = all_candidates[0][0] if all_candidates else beam[0][0]
            return best_seq, attention_weights