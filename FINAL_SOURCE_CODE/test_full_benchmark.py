import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from evaluate import grid,ssim
from full_benchmark import FACTORIES,save_state
from models import DDPM
from run import losses


class FullBenchmarkChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_six_losses_have_finite_nonzero_gradients(self):
        torch.manual_seed(42)
        x=torch.rand(4,1,28,28)
        y=torch.arange(4)
        for name,factory in FACTORIES.items():
            with self.subTest(model=name):
                model=factory()
                if name in ("GAN","DCGAN"):
                    opts=[torch.optim.Adam(net.parameters()) for net in (model.g,model.d)]
                    initial=next(model.d.parameters()).detach().clone()
                    value,_=losses(name,model,x,y,None,opts)
                    self.assertFalse(torch.equal(initial,next(model.d.parameters())))
                    net=model.g
                    self.assertEqual(model.g(torch.randn(4,model.z_dim)).shape,x.shape)
                else:
                    value,_=losses(name,model,x,y,None)
                    net=model
                    if name=="DDPM":
                        self.assertEqual(model(x,y).shape,x.shape)
                    else:
                        self.assertEqual(model(x)[0].shape,x.shape)
                value.backward()
                grads=[p.grad for p in net.parameters() if p.grad is not None]
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(all(g.isfinite().all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads),0.)

    def test_checkpoint_and_figure_roundtrip(self):
        torch.manual_seed(42)
        images=torch.rand(2,1,28,28)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path=Path(tmp)
            model=FACTORIES["AE"]().eval()
            save_state(path/"AE.pt",model,1,.25)
            restored=FACTORIES["AE"]().eval()
            ckpt=torch.load(path/"AE.pt",weights_only=True)
            restored.load_state_dict(ckpt["state"])
            with torch.inference_mode():
                pred=restored(images)[0]
                torch.testing.assert_close(pred,model(images)[0])
                self.assertTrue(ssim(images,pred).isfinite().all())
            grid(path/"actual.png",[images,pred],["Real","AE"])
            with Image.open(path/"actual.png") as im:
                im.verify()
            self.assertEqual(ckpt["epoch"],1)


if __name__=="__main__":
    unittest.main()
