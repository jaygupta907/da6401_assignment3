import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_dim, hidden_dim, num_layers=1, rnn_type='LSTM', bidirectional=True, dropout=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.rnn_type = rnn_type
        rnn_cls = getattr(nn, rnn_type)  # nn.LSTM, nn.GRU, or nn.RNN
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        # input_seq: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(input_seq))  # Apply dropout to embeddings
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden  # hidden: (num_layers*dirs, batch, hidden_dim) or tuple

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_dim, hidden_dim, num_layers=1, rnn_type='LSTM', bidirectional=False, dropout=0.2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.rnn_type = rnn_type
        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden):
        # input_token: (batch_size, 1)
        embedded = self.dropout(self.embedding(input_token))  # Apply dropout to embeddings
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))  # (batch, vocab_size)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, rnn_type='LSTM'):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_type = rnn_type

    def translate(self, input_tensor, output_vocab, max_len=30, beam_width=1):
        self.eval()
        with torch.no_grad():
            # Encode the input sequence
            _ , encoder_hidden = self.encoder(input_tensor)

            # Initialize the beam
            trg_indexes = [[output_vocab.char2idx['<sos>']]]  # Start with <sos>
            beam_scores = torch.zeros(1, device=input_tensor.device)  # Initial score for the beam

            # Combine bidirectional hidden states for the Decoder
            if self.rnn_type == 'LSTM':
                decoder_hidden = (self._combine_directions(encoder_hidden[0]), self._combine_directions(encoder_hidden[1]))
            else:
                decoder_hidden = self._combine_directions(encoder_hidden)

            # Iterate through the maximum sequence length
            for _ in range(max_len):
                all_candidates = []  # Store all candidates for the current step

                # Iterate through each sequence in the beam
                for i, seq in enumerate(trg_indexes):
                    # Stop expanding sequences that already ended with <eos>
                    if seq[-1] == output_vocab.char2idx['<eos>']:
                        all_candidates.append((seq, beam_scores[i]))
                        continue

                    # Prepare the input token for the decoder
                    trg_tensor = torch.tensor([seq[-1]], device=input_tensor.device).unsqueeze(0)

                    # Decode the next token
                    output, decoder_hidden = self.decoder(trg_tensor, decoder_hidden)
                    output = F.log_softmax(output, dim=1)  # Apply log softmax for probabilities

                    # Get the top `beam_width` tokens and their scores
                    topk_scores, topk_indices = output.topk(beam_width)

                    # Add each candidate to the list
                    for k in range(beam_width):
                        candidate_seq = seq + [topk_indices[0, k].item()]
                        candidate_score = beam_scores[i] + topk_scores[0, k]
                        all_candidates.append((candidate_seq, candidate_score))

                # Sort all candidates by score and keep the top `beam_width`
                all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

                # Update the beam with the top candidates
                trg_indexes = [candidate[0] for candidate in all_candidates]
                beam_scores = torch.tensor([candidate[1] for candidate in all_candidates], device=input_tensor.device)

            # Return the sequence with the highest score
            best_sequence = trg_indexes[0]
            return best_sequence

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        output_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, output_vocab_size, device=trg.device)

        # Encode the source sequence
        _, hidden = self.encoder(src)

        # Combine bidirectional hidden states for the Decoder
        if self.rnn_type == 'LSTM':
            hidden = (self._combine_directions(hidden[0]), self._combine_directions(hidden[1]))
        else:
            hidden = self._combine_directions(hidden)

        input_token = trg[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    def _combine_directions(self, hidden):
        return hidden.view(self.decoder.rnn.num_layers, 2, hidden.size(1), hidden.size(2)).sum(1)