import torch
from torch import nn
from torch.nn import functional as F


def decoder():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid(),
                         nn.Unflatten(1, (1, 28, 28)))


class AE(nn.Module):
    def __init__(self, vae=False):
        super().__init__()
        self.vae = vae
        self.hidden = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mean = nn.Linear(256, 64)
        if vae:
            self.logvar = nn.Linear(256, 64)
        self.decode = decoder()

    def encode(self, x):
        h = self.hidden(x)
        z = self.mean(h)
        return (z, self.logvar(h).clamp(-12, 8)) if self.vae else (F.relu(z), None)

    def forward(self, x):
        z, lv = self.encode(x)
        sample = z + (lv * .5).exp() * torch.randn_like(z) if self.vae and self.training else z
        return self.decode(sample), z, lv


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = nn.Linear(784, 256), nn.Linear(256, 64), nn.Linear(64, 10)

    def encode(self, x):
        return F.relu(self.b(F.relu(self.a(x.flatten(1)))))

    def forward(self, x):
        return self.c(self.encode(x))

    def transpose_latent(self, z):
        return torch.sigmoid(F.relu(z @ self.b.weight) @ self.a.weight).reshape(-1, 1, 28, 28)

    def transpose_class(self, y):
        return self.transpose_latent(F.relu(F.one_hot(y, 10).float() @ self.c.weight))


class Adversary(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.z_dim = 100 if conv else 64
        if conv:
            self.g = nn.Sequential(nn.Unflatten(1, (self.z_dim, 1, 1)),
                                   nn.ConvTranspose2d(self.z_dim, 128, 7, bias=False),
                                   nn.BatchNorm2d(128), nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                   nn.BatchNorm2d(64), nn.ReLU(), nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid())
            self.d = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(.2),
                                   nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(.2),
                                   nn.Conv2d(128, 1, 7), nn.Flatten(1))
        else:
            self.g = decoder()
            self.d = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.LeakyReLU(.2),
                                   nn.Linear(256, 1))


class DDPM(nn.Module):
    def __init__(self, steps=1000):
        super().__init__()
        self.steps = steps
        beta = torch.linspace(.0001, .02, steps)
        bar = (1-beta).cumprod(0)
        prev = F.pad(bar[:-1], (1, 0), value=1.)
        for name, value in dict(beta=beta, bar=bar, var=beta*(1-prev)/(1-bar)).items():
            self.register_buffer(name, value)
        self.time = nn.Embedding(steps, 32)
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.SiLU())
        self.down = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.up = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.out = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, x, t):
        h = self.enc(x) + self.time(t)[:, :, None, None]
        return self.out(torch.cat([h, self.up(self.down(h))], 1))

    def loss(self, x):
        t = torch.randint(self.steps, (len(x),), device=x.device)
        a = self.bar[t, None, None, None]
        eps = torch.randn_like(x)
        return F.mse_loss(self(a.sqrt()*(2*x-1) + (1-a).sqrt()*eps, t), eps)

    @torch.no_grad()
    def sample(self, n, trace=None):
        x = torch.randn(n, 1, 28, 28, device=self.beta.device)
        if trace is not None:
            trace.append((self.steps, ((x[:1]+1)/2).clamp(0, 1).cpu()))
        for k in reversed(range(self.steps)):
            t = torch.full((n,), k, device=x.device, dtype=torch.long)
            b = self.beta[k]
            x = (x - b*self(x, t)/(1-self.bar[k]).sqrt())/(1-b).sqrt()
            if k:
                x = x + self.var[k].sqrt()*torch.randn_like(x)
            if trace is not None and k in {0, self.steps//4, self.steps//2, 3*self.steps//4, self.steps-1}:
                trace.append((k, ((x[:1]+1)/2).clamp(0, 1).cpu()))
        return ((x+1)/2).clamp(0, 1)
