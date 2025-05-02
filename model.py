import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_dim, hidden_dim, num_layers=1, rnn_type='LSTM', bidirectional=True):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.rnn_type = rnn_type
        rnn_cls = getattr(nn, rnn_type)  # nn.LSTM, nn.GRU, or nn.RNN
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=bidirectional)

    def forward(self, input_seq):
        # input_seq: (batch_size, seq_len)
        embedded = self.embedding(input_seq)  # (batch, seq_len, embed_dim)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden  # hidden: (num_layers*dirs, batch, hidden_dim) or tuple

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_dim, hidden_dim, num_layers=1, rnn_type='LSTM', bidirectional=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.rnn_type = rnn_type
        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=bidirectional)
        # hidden_dim* (2 if bidirectional else 1)
        self.fc_out = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_vocab_size)

    def forward(self, input_token, hidden):
        # input_token: (batch_size, 1)
        embedded = self.embedding(input_token)  # (batch, 1, embed_dim)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))  # (batch, vocab_size)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, rnn_type='LSTM'):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_type = rnn_type

    def translate(self, input_tensor, output_vocab, max_len=30):
        self.eval()
        with torch.no_grad():
            _, encoder_hidden = self.encoder(input_tensor)
            trg_indexes = [output_vocab.char2idx['<sos>']]
            for _ in range(max_len):
                trg_tensor = torch.tensor([trg_indexes[-1]], device=input_tensor.device).unsqueeze(0)
                output, encoder_hidden = self.decoder(trg_tensor, encoder_hidden)
                _, topi = output.topk(1)
                next_idx = topi.item()
                trg_indexes.append(next_idx)
                if next_idx == output_vocab.char2idx['<eos>']:
                    break
        return output_vocab.decode(trg_indexes)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        output_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, output_vocab_size, device=trg.device)

        _, hidden = self.encoder(src)
        if self.rnn_type == 'LSTM':
            decoder_hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            decoder_hidden = hidden.detach()

        input_token = trg[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            output, decoder_hidden = self.decoder(input_token, decoder_hidden)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs