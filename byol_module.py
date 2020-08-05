import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision.models import densenet
import math

from copy import deepcopy

import pl_bolts
from pl_bolts import metrics
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule
from pl_bolts.losses.self_supervised_learning import nt_xent_loss
from pl_bolts.metrics import mean
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pl_bolts.optimizers.layer_adaptive_scaling import LARS
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


class DensenetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = densenet.densenet121(pretrained=False, num_classes=1)
        del self.model.classifier

    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()

        if encoder is None:
            encoder = torchvision_ssl_encoder('resnet50')
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP()
        # Predictor
        self.predictor = MLP(input_dim=256)

    def forward(self, x):
        y = self.encoder(x)[0]
        y = y.view(y.size(0), -1)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


class BYOL(pl.LightningModule):
    def __init__(self,
                 datamodule: pl.LightningDataModule = None,
                 data_dir: str = './',
                 learning_rate: float = 0.00006,
                 weight_decay: float = 0.0005,
                 input_height: int = 32,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 optimizer: str = 'lars',
                 lr_sched_step: float = 30.0,
                 lr_sched_gamma: float = 0.5,
                 lars_momentum: float = 0.9,
                 lars_eta: float = 0.001,
                 loss_temperature: float = 0.5,
                 **kwargs):
        """
        .. warning:: Work in progress. This implementation is still being verified

        PyTorch Lightning implementation of `BYOL <https://arxiv.org/pdf/2006.07733.pdf.>`_
        Paper authors: Jean-Bastien Grill ,Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond,
        Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar,
        Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko.

        Model implemented by:
            - `Annika Brundyn <https://github.com/annikabrundyn>`_

        Example:
            >>> from pl_bolts.models.self_supervised import BYOL
            ...
            >>> model = BYOL()
        Train::
            trainer = Trainer()
            trainer.fit(model)
        CLI command::
            # cifar10
            python byol_module.py --gpus 1
            # imagenet
            python byol_module.py
                --gpus 8
                --dataset imagenet2012
                --data_dir /path/to/imagenet/
                --meta_dir /path/to/folder/with/meta.bin/
                --batch_size 32
        Args:
            datamodule: The datamodule
            data_dir: directory to store data
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            optimizer: optimizer name
            lr_sched_step: step for learning rate scheduler
            lr_sched_gamma: gamma for learning rate scheduler
            lars_momentum: the mom param for lars optimizer
            lars_eta: for lars optimizer
            loss_temperature: float = 0.
        """
        super().__init__()
        self.save_hyperparameters()

        # init default datamodule
        if datamodule is None:
            datamodule = CIFAR10DataModule(data_dir, num_workers=num_workers, batch_size=batch_size)
            datamodule.train_transforms = SimCLRTrainDataTransform(input_height)
            datamodule.val_transforms = SimCLREvalDataTransform(input_height)

        self.datamodule = datamodule

        self.online_network = SiameseArm()
        self.target_network = deepcopy(self.online_network)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shared_step(self, batch, batch_idx):
        (img_1, img_2), y = batch

        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        loss_a = F.mse_loss(h1, z2)

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        loss_b = F.mse_loss(h1, z2)

        # final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        result = pl.TrainResult(minimize=total_loss)
        result.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'train_loss': total_loss})

        return result

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        result = pl.EvalResult(early_stop_on=total_loss, checkpoint_on=total_loss)
        result.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'train_loss': total_loss})

        return result

    def configure_optimizers(self):
        optimizer = LARS(self.parameters(), lr=self.hparams.learning_rate)
        # TODO: add scheduler
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--online_ft', action='store_true', help='run online finetuner')
        parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, imagenet2012, stl10')

        (args, _) = parser.parse_known_args()
        # Data
        parser.add_argument('--data_dir', type=str, default='.')

        # Training
        parser.add_argument('--optimizer', choices=['adam', 'lars'], default='lars')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=1.0)
        parser.add_argument('--lars_momentum', type=float, default=0.9)
        parser.add_argument('--lars_eta', type=float, default=0.001)
        parser.add_argument('--lr_sched_step', type=float, default=30, help='lr scheduler step')
        parser.add_argument('--lr_sched_gamma', type=float, default=0.5, help='lr scheduler step')
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        # Model
        parser.add_argument('--loss_temperature', type=float, default=0.5)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--meta_dir', default='.', type=str, help='path to meta.bin for imagenet')

        return parser


class BYOLMAWeightUpdate(pl.Callback):

    def __init__(self, initial_tau=0.996):
        """
        Weight update rule from BYOL.
        Your model should have a:
            - self.online_network
            - self.target_network

        Updates the target_network params using an exponential moving average update rule weighted by tau.
        BYOL claims this keeps the online_network from collapsing.

        Arg:
            initial_tau: starting tau. Auto-updates with every training step
        """
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_batch_end(self, trainer, pl_module):
        # get networks
        online_net = pl_module.online_network
        target_net = pl_module.target_network

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(pl_module)

    def update_tau(self, pl_module):
        tau = 1 - (1 - self.initial_tau) * (torch.cos(math.pi * pl_module.global_step / trainer.max_steps) + 1) / 2
        return tau

    def update_weights(self, online_net, target_net):
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(online_net.named_parameters(), target_net.named_parameters()):
            if 'weight' in name:
                target_p.data = self.current_tau * target_p.data + (1 - self.current_tau) * online_p.data


def test_byol_ma_weight_update_callback():
    a = nn.Linear(100, 10)
    b = deepcopy(a)
    a_original = deepcopy(a)
    b_original = deepcopy(b)

    # make sure a params and b params are the same
    assert torch.equal(next(iter(a.parameters()))[0], next(iter(b.parameters()))[0])

    # fake weight update
    opt = torch.optim.SGD(a.parameters(), lr=0.1)
    y = a(torch.randn(3, 100))
    loss = y.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    # make sure a did in fact update
    assert not torch.equal(next(iter(a_original.parameters()))[0], next(iter(a.parameters()))[0])

    # do update via callback
    cb = BYOLMAWeightUpdate(0.8)
    cb.update_weights(a, b)

    assert not torch.equal(next(iter(b_original.parameters()))[0], next(iter(b.parameters()))[0])


# todo: covert to CLI func and add test
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = BYOL.add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    datamodule = None
    if args.dataset == 'stl10':
        datamodule = STL10DataModule.from_argparse_args(args)
        datamodule.train_dataloader = datamodule.train_dataloader_mixed
        datamodule.val_dataloader = datamodule.val_dataloader_mixed

        (c, h, w) = datamodule.size()
        datamodule.train_transforms = SimCLRTrainDataTransform(h)
        datamodule.val_transforms = SimCLREvalDataTransform(h)

    elif args.dataset == 'imagenet2012':
        datamodule = ImagenetDataModule.from_argparse_args(args, image_size=196)
        (c, h, w) = datamodule.size()
        datamodule.train_transforms = SimCLRTrainDataTransform(h)
        datamodule.val_transforms = SimCLREvalDataTransform(h)

    model = BYOL(**args.__dict__, datamodule=datamodule)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[BYOLMAWeightUpdate()])
    trainer.fit(model)