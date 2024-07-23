import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser(description='train transformer por material segmentation.')
    parser.add_argument('--exp_name', type=str, help='Experiment name', required=True)
    
    parser.add_argument('--model', type=str, help='Model to train', required=True)
    
    parser.add_argument('--datadir', type=str, help='Path to the dataset', required=True)
    
    parser.add_argument('--gpu', type=int, help='GPU number', required=True)
    
    parser.add_argument('--epochs', type=int, help='Number of epochs', required=True)
    
    parser.add_argument('--batch_size', type=int, help='Batch size', required=True)
    
    parser.add_argument('--lr', type=float, help='Learning r    ate', required=True)
    
    parser.add_argument('--wandb', default=False, action=argparse.BooleanOptionalAction, help='Use wandb')
    
    parser.add_argument('--usergb', default=False, action=argparse.BooleanOptionalAction, help='Use rgb as input')
    
    parser.add_argument('--repeatrgb', default=False, action=argparse.BooleanOptionalAction, help='Repeat rgb as input')

    # reconstruct cube arguments
    parser.add_argument('--reconstruct', default=False, action=argparse.BooleanOptionalAction, help='Reconstruct cube')
    
    parser.add_argument('--regularize', default=False, action=argparse.BooleanOptionalAction, help='Regularize cube')


    parser.add_argument('--weights', type=str, help='Path to model weights', required=False)
    
    parser.add_argument('--group', type=str, help='Group to use', default=None, required=False)
    
    return parser
