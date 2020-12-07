import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--lr', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    args = parser.parse_args()
    return args
