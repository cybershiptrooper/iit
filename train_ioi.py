from iit.utils.train_scripts import train_ioi
from iit.utils.io_scripts import save_model
from iit.utils.argparsing import IOIArgParseNamespace
import torch as t

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-iit", type=float, default=1.0)
    parser.add_argument("-b", type=float, default=1.0)
    parser.add_argument("-s", type=float, default=0.4)
    parser.add_argument("--next-token", action="store_true")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("-single-loss", "--use-single-loss", action="store_true")
    parser.add_argument("--save-to-wandb", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--include-mlp", action="store_true")

    args = parser.parse_args()
    namespace = IOIArgParseNamespace(**vars(args))

    model_pair = train_ioi(namespace)
    save_model(model_pair, namespace, "ioi")