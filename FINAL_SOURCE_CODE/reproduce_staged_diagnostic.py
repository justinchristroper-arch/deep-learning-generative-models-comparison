"""Replay the original quick training RNG sequence, then refresh_dcgan.py replaces its prototype."""
import sys

from torch import nn

from models import Adversary
from run import main


class ProjectionPrototype(nn.Module):
    """Archived initial network for replaying the supplied run's random-number stream."""
    def __init__(self):
        super().__init__()
        self.z_dim = 100
        self.g = nn.Sequential(nn.Linear(100,128*7*7),nn.Unflatten(1,(128,7,7)),
                               nn.BatchNorm2d(128),nn.ReLU(),nn.ConvTranspose2d(128,64,4,2,1),
                               nn.BatchNorm2d(64),nn.ReLU(),nn.ConvTranspose2d(64,1,4,2,1),nn.Sigmoid())
        self.d = nn.Sequential(nn.Conv2d(1,64,4,2,1),nn.LeakyReLU(.2),
                               nn.Conv2d(64,128,4,2,1),nn.LeakyReLU(.2),
                               nn.Flatten(),nn.Linear(128*7*7,1))


def historical_adversary(conv):
    return ProjectionPrototype() if conv else Adversary(False)


if __name__ == "__main__":
    if any(flag in sys.argv for flag in ("--full","--smoke","--classifier-only")):
        raise SystemExit("This replay is exclusively for the five-epoch quick diagnostic")
    if "--quick" not in sys.argv:
        sys.argv.append("--quick")
    if "--epochs" not in sys.argv and not any(v.startswith("--epochs=") for v in sys.argv):
        sys.argv.extend(["--epochs","5"])
    if "--out" not in sys.argv and not any(v.startswith("--out=") for v in sys.argv):
        sys.argv.extend(["--out","runs/replayed_diagnostic"])
    main(adversary_factory=historical_adversary,fit_only=True)
