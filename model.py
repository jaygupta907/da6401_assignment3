import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_dim, hidden_dim, num_layers=1, rnn_type='LSTM'):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.rnn_type = rnn_type
        rnn_cls = getattr(nn, rnn_type)  # nn.LSTM, nn.GRU, or nn.RNN
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, input_seq):
        # input_seq: (batch_size, seq_len)
        embedded = self.embedding(input_seq)  # (batch_size, seq_len, embed_dim)
        outputs, hidden = self.rnn(embedded)  # outputs: (batch_size, seq_len, hidden_dim)
        return outputs, hidden  # hidden: (num_layers, batch_size, hidden_dim) or tuple if LSTM


class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_dim, hidden_dim, num_layers=1, rnn_type='LSTM'):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.rnn_type = rnn_type
        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, input_token, hidden):
        # input_token: (batch_size, 1)
        embedded = self.embedding(input_token)  # (batch_size, 1, embed_dim)
        output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, 1, hidden_dim)
        prediction = self.fc_out(output.squeeze(1))  # (batch_size, vocab_size)
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, rnn_type='LSTM'):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_type = rnn_type

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        output_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, output_vocab_size).to(trg.device)

        encoder_outputs, hidden = self.encoder(src)

        if self.rnn_type == 'LSTM':
            decoder_hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            decoder_hidden = hidden.detach()

        input_token = trg[:, 0].unsqueeze(1)  # <sos> token

        for t in range(1, trg_len):
            output, decoder_hidden = self.decoder(input_token, decoder_hidden)
            outputs[:, t] = output
            top1 = output.argmax(1).unsqueeze(1)
            input_token = trg[:, t].unsqueeze(1) if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
