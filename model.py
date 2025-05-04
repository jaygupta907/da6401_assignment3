import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, decoder_hidden_dim, encoder_output_dim):
        super(Attention, self).__init__()
        # Linear layers to project decoder hidden state and encoder outputs
        self.attn_W1 = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.attn_W2 = nn.Linear(encoder_output_dim, decoder_hidden_dim)
        # Layer to compute attention scores
        self.attn_v = nn.Linear(decoder_hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch_size, decoder_hidden_dim)
        # encoder_outputs: (batch_size, seq_len, encoder_output_dim)
        enc_proj = self.attn_W2(encoder_outputs)  # (batch_size, seq_len, decoder_hidden_dim)
        dec_proj = self.attn_W1(decoder_hidden).unsqueeze(1)  # (batch_size, 1, decoder_hidden_dim)
        attn_input = torch.tanh(dec_proj + enc_proj)  # (batch_size, seq_len, decoder_hidden_dim)
        scores = self.attn_v(attn_input).squeeze(2)  # (batch_size, seq_len)
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, encoder_output_dim)
        return context,attn_weights



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

    def forward(self, input_seq):
        embedded = self.dropout(self.embedding(input_seq))  # (batch_size, seq_len, embed_dim)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden  # outputs: (batch_size, seq_len, hidden_dim * num_directions)

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_dim, hidden_dim, encoder_output_dim, use_attention=False, num_layers=1, rnn_type='LSTM', bidirectional=False, dropout=0.2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.rnn_type = rnn_type
        self.use_attention = use_attention
        self.encoder_output_dim = encoder_output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initialize Attention module if enabled
        if use_attention:
            self.attention = Attention(hidden_dim, encoder_output_dim)
            rnn_input_size = embed_dim + encoder_output_dim  # Concatenate embedding and context
        else:
            self.attention = None
            rnn_input_size = embed_dim  # Only embedding

        # RNN setup
        rnn_cls = getattr(nn, rnn_type)  # nn.LSTM, nn.GRU, or nn.RNN
        self.rnn = rnn_cls(rnn_input_size, hidden_dim, num_layers,
                           batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs):
        # input_token: (batch_size, 1)
        embedded = self.dropout(self.embedding(input_token)).squeeze(1)  # (batch_size, embed_dim)

        if self.use_attention:
            # Extract hidden state for attention (last layer)
            if self.rnn_type == 'LSTM':
                h_att = hidden[0][-1]  # (batch_size, hidden_dim)
            else:
                h_att = hidden[-1]  # (batch_size, hidden_dim)
            context ,_ = self.attention(h_att, encoder_outputs)  # (batch_size, encoder_output_dim)
            rnn_input = torch.cat([embedded, context], dim=1).unsqueeze(1)  # (batch_size, 1, embed_dim + encoder_output_dim)
        else:
            rnn_input = embedded.unsqueeze(1)  # (batch_size, 1, embed_dim)

        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(output.squeeze(1))  # (batch_size, output_vocab_size)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, rnn_type='LSTM'):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_type = rnn_type

    def translate(self, input_tensor, output_vocab, max_len=30, beam_width=1):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for inference
            # Encode the input sequence
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)

            # Initialize the beam with the start-of-sequence token
            trg_indexes = [[output_vocab.char2idx['<sos>']]]
            beam_scores = torch.zeros(1, device=input_tensor.device)

            # Prepare the decoder's initial hidden state by combining bidirectional encoder outputs
            if self.rnn_type == 'LSTM':
                decoder_hidden = (
                    self._combine_directions(encoder_hidden[0]),
                    self._combine_directions(encoder_hidden[1])
                )
            else:
                decoder_hidden = self._combine_directions(encoder_hidden)

            # Store attention weights for visualization
            attention_weights = []

            # Generate the output sequence up to max_len
            for _ in range(max_len):
                all_candidates = []
                for i, seq in enumerate(trg_indexes):
                    # If the sequence has ended with <eos>, keep it as a candidate
                    if seq[-1] == output_vocab.char2idx['<eos>']:
                        all_candidates.append((seq, beam_scores[i]))
                        continue
                    
                    # Prepare the last token as input to the decoder
                    trg_tensor = torch.tensor([seq[-1]], device=input_tensor.device).unsqueeze(0)
                    
                    # Pass encoder_outputs to the decoder; it will be used only if attention is enabled
                    output, decoder_hidden = self.decoder(trg_tensor, decoder_hidden, encoder_outputs)
                    
                    # Collect attention weights if attention is enabled
                    if self.decoder.use_attention:
                        h_att = decoder_hidden[0][-1] if self.rnn_type == 'LSTM' else decoder_hidden[-1]
                        context,attn_weights = self.decoder.attention(h_att, encoder_outputs)
                        attention_weights.append(attn_weights.cpu().numpy())
                    
                    # Apply log softmax to get log probabilities
                    output = F.log_softmax(output, dim=1)
                    
                    # Get the top beam_width predictions
                    topk_scores, topk_indices = output.topk(beam_width)
                    
                    # Create candidates for each top prediction
                    for k in range(beam_width):
                        candidate_seq = seq + [topk_indices[0, k].item()]
                        candidate_score = beam_scores[i] + topk_scores[0, k]
                        all_candidates.append((candidate_seq, candidate_score))
                
                # Sort candidates by score and select the top beam_width
                all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                trg_indexes = [candidate[0] for candidate in all_candidates]
                beam_scores = torch.tensor([candidate[1] for candidate in all_candidates], 
                                        device=input_tensor.device)
            
            # Return the best sequence (first in the beam) and attention weights
            return trg_indexes[0], attention_weights

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        output_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, output_vocab_size, device=trg.device)

        encoder_outputs, hidden = self.encoder(src)
        # Combine bidirectional hidden states
        if self.rnn_type == 'LSTM':
            hidden = (self._combine_directions(hidden[0]), self._combine_directions(hidden[1]))
        else:
            hidden = self._combine_directions(hidden)

        input_token = trg[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    def _combine_directions(self, hidden):
        return hidden.view(self.decoder.rnn.num_layers, 2, hidden.size(1), hidden.size(2)).sum(1)