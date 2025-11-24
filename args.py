import argparse

def get_args():
    parser = argparse.ArgumentParser(description="TCN RUL Training Config")

    parser.add_argument("--dataset_dir", type=str, default="/Users/floimb/Documents/data/CMaps")
    parser.add_argument("--set_number", type=int, default=4)
    parser.add_argument("--model_dir", type=str, default="/Users/floimb/Documents/Models/RUL")
    parser.add_argument("--model_name", type=str, default="tcn_rul")

    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--rul_clip", type=int, default=125)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1000)

    parser.add_argument("--num_filters", type=int, default=200)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--lr_decay_patience", type=int, default=20)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    return args