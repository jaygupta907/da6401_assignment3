import argparse 

def get_args():
    """
    Parses command-line arguments and returns them.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Convolutional Neural Network')

    # Adding various arguments for training configuration
    parser.add_argument('--wandb_project', type=str, default='Convolution Neural Networks', help='Project Name')
    parser.add_argument('--wandb_entity', type=str, default='jay_gupta-indian-institute-of-technology-madras', help='WandB entity name')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--input_embedding', type=int, default=512, help='Input embedding size')
    parser.add_argument('--encoder_layers',type=int,default=2,help='Number of encoder layers')
    parser.add_argument('--decoder_layers',type=int,default=2,help="Number of decoder layers")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of the optimizer')
    parser.add_argument('--hidden_layer_size',type=float,default=512,help="Hidden layer size")
    parser.add_argument('--cell_type',type=str,default='RNN',help='Type of RNN cell to use (LSTM, GRU, etc.)')
    parser.add_argument('--dropout',type=str,default=0.2,help='Dropout probability in dense layers')
    parser.add_argument('--encoder_bidirectional',type=bool,default=True,help='Use bidirectional RNN for encoder')
    parser.add_argument('--decoder_bidirectional',type=bool,default=False,help='Use bidirectional RNN for decoder')
    parser.add_argument('--use_attention',type=bool,default=False,help='Use attention mechanism')
    parser.add_argument('--beam_size', type=float, default=1, help='Beam size for beam search')
    return parser.parse_args()