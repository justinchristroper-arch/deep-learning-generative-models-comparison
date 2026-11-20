import unittest
import tempfile
from pathlib import Path
import torch
from torch.nn import functional as F

from evaluate import ssim
from models import AE, Adversary, Classifier, DDPM, decoder
from data import loaders
from run import losses, weight_hash


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_transpose_orientation(self):
        m = Classifier()
        y = torch.tensor([2, 5])
        z = F.relu(F.linear(F.one_hot(y, 10).float(), m.c.weight.T))
        h = F.relu(F.linear(z, m.b.weight.T))
        expected = F.linear(h, m.a.weight.T).sigmoid().reshape(2,1,28,28)
        torch.testing.assert_close(expected, m.transpose_class(y))

    def test_ssim_identity(self):
        x = torch.rand(5,1,28,28)
        torch.testing.assert_close(ssim(x,x), torch.ones(5))

    def test_dcgan_is_fully_convolutional(self):
        m = Adversary(True)
        self.assertFalse(any(isinstance(layer, torch.nn.Linear) for layer in m.modules()))
        images = m.g(torch.randn(2,100))
        self.assertEqual(images.shape, (2,1,28,28))
        self.assertEqual(m.d(images).shape, (2,1))

    def test_vae_eval_uses_posterior_mean(self):
        m = AE(vae=True).eval()
        x = torch.rand(2,1,28,28)
        pred, z, _ = m(x)
        torch.testing.assert_close(pred, m.decode(z))
        torch.testing.assert_close(pred, m(x)[0])

    def test_diffusion_visits_every_timestep(self):
        m = DDPM(8)
        steps = []
        token = m.register_forward_pre_hook(lambda module, args: steps.append(args[1][0].item()))
        x = m.sample(2)
        token.remove()
        self.assertEqual(steps, list(reversed(range(8))))
        self.assertEqual(tuple(x.shape), (2,1,28,28))
        self.assertTrue(torch.all((0 <= x) & (x <= 1)))
        self.assertEqual(m.var[0].item(), 0.)

    def test_split_membership_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            loaders(out, out, "smoke")
            ix = torch.load(out/"indices.pt", weights_only=True)
            joined = torch.cat(list(ix.values()))
            self.assertEqual([len(v) for v in ix.values()], [256,32,32])
            torch.testing.assert_close(joined.sort().values, torch.arange(320))
            torch.testing.assert_close(joined, torch.randperm(320, generator=torch.Generator().manual_seed(42)))

    def test_decoder_learning_preserves_classifier(self):
        clf = Classifier().eval().requires_grad_(False)
        d = decoder()
        opt = torch.optim.Adam(d.parameters())
        original = weight_hash(clf)
        initial_decoder = weight_hash(d)
        x = torch.rand(4,1,28,28)
        loss,_ = losses("LatentDecoder", d, x, None, clf)
        loss.backward()
        opt.step()
        self.assertEqual(original, weight_hash(clf))
        self.assertNotEqual(initial_decoder, weight_hash(d))


if __name__ == "__main__":
    unittest.main()
