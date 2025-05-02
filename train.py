import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DakshinaDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq
import logging
from tqdm import tqdm
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_accuracy(predictions, target):
    """Calculate accuracy for each batch"""
    _, predicted_ids = predictions.max(dim=1)
    correct = (predicted_ids == target).sum().item()
    total = target.size(0)
    accuracy = correct / total
    return accuracy

def train_one_epoch(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for src, trg, _, _ in progress_bar:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # Calculate accuracy
        accuracy = calculate_accuracy(output, trg)

        epoch_loss += loss.item()
        epoch_accuracy += accuracy

        progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy)

    return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for src, trg, _, _ in progress_bar:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            # Calculate accuracy
            accuracy = calculate_accuracy(output, trg)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy

            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy)

    return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)

def main():
    train_path = "datasets/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    dev_path = "datasets/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"

    train_dataset = DakshinaDataset(train_path)
    dev_dataset = DakshinaDataset(dev_path)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    input_vocab_size = len(train_dataset.input_vocab)
    output_vocab_size = len(train_dataset.output_vocab)

    encoder = Encoder(input_vocab_size, embed_dim=64, hidden_dim=128, num_layers=1, rnn_type='LSTM')
    decoder = Decoder(output_vocab_size, embed_dim=64, hidden_dim=128, num_layers=1, rnn_type='LSTM')
    model = Seq2Seq(encoder, decoder, rnn_type='LSTM').to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad token is index 0

    N_EPOCHS = 10
    CLIP = 1

    for epoch in range(N_EPOCHS):
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS}")
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, CLIP)
        print("="*os.get_terminal_size().columns)
        dev_loss, dev_accuracy = evaluate(model, dev_loader, criterion)

        logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f} ")

if __name__ == '__main__':
    main()
