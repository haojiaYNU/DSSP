import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import MaskDataset, get_dataset
from model import UnlabeledLoss, VIMESelf, VIMESemi
from utils import EarlyStopping, mask_generator, perf_metric, pretext_generator

log = logging.getLogger(__name__)


class Train:
    def __init__(self, config) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.l_set, self.u_set, self.test_set = get_dataset(config['data_name'], config['label_data_rate'])
        self.self_epochs = config['self_epochs']
        self.semi_max_iter = config['semi_max_iter']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.k = config['k']
        self.p_m = config['p_m']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.l_samples = len(self.l_set)
        self.dim = self.l_set[0][0].shape[-1]
        self.l_dim = self.l_set[0][1].shape[-1]

        self.vime_self = VIMESelf(self.dim, self.dim).to(self.device)
        self.vime_semi = VIMESemi(self.dim, self.l_dim).to(self.device)

        self.l_loss_fn = nn.CrossEntropyLoss()
        self.u_loss_fn = UnlabeledLoss()

        self.opt_self = optim.RMSprop(self.vime_self.parameters(), lr=1e-3)
        self.opt_semi = optim.Adam(self.vime_semi.parameters())

        self.scheduler = StepLR(self.opt_self, step_size=50, gamma=0.1)
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])

    def semi_sl(self):
        idx = np.random.permutation(len(self.l_set))
        train_idx = idx[:int(len(idx) * 0.9)]
        valid_idx = idx[int(len(idx) * 0.9):]

        val_set = self.l_set[valid_idx]

        for i in range(self.semi_max_iter):
            b_idx = np.random.permutation(len(train_idx))[:self.batch_size].tolist()
            l_batch = self.l_set[train_idx[b_idx]]
            x_batch, y_batch = l_batch
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                x_batch = self.vime_self.encoder(x_batch)

            bu_idx = np.random.permutation(len(self.u_set))[:self.batch_size]
            u_batch = self.u_set[bu_idx]
            xu_batch_ori, _ = u_batch
            xu_batch_ori = xu_batch_ori.to(self.device)

            xu_batch = []
            for _ in range(self.k):
                m_batch = mask_generator(xu_batch_ori.shape, self.p_m)
                m_batch = m_batch.to(self.device)
                _, xu_batch_temp = pretext_generator(m_batch, xu_batch_ori)

                with torch.no_grad():
                    xu_batch_temp = self.vime_self.encoder(xu_batch_temp)
                xu_batch.append(xu_batch_temp.unsqueeze(0))
            xu_batch = torch.cat(xu_batch)

            self.opt_semi.zero_grad()
            l_pred = self.vime_semi(x_batch)
            u_pred = self.vime_semi(xu_batch)

            l_loss = self.l_loss_fn(l_pred, y_batch)
            u_loss = self.u_loss_fn(u_pred)
            loss = l_loss + self.beta * u_loss
            loss.backward()
            self.opt_semi.step()

            with torch.no_grad():
                x, y = val_set
                x = x.to(self.device)
                y = y.to(self.device)
                x = self.vime_self.encoder(x)
                pred = self.vime_semi(x)
                val_loss = self.l_loss_fn(pred, y)

            if i % 100 == 0:
                print(f'Iteration: {i} / {self.semi_max_iter}, '
                      f'Current loss (val): {val_loss.item(): .4f}, '
                      f'Current loss (train): {loss.item(): .4f}, '
                      f'supervised loss: {l_loss.item(): .4f}, '
                      f'unsupervised loss: {u_loss.item(): .4f}')

            if i % math.ceil(self.l_samples / self.batch_size) == 0:
                self.early_stopping(val_loss, self.vime_semi)
                if self.early_stopping.early_stop:
                    print(f'early stopping {i} / {self.semi_max_iter}')
                    self.vime_semi.load_state_dict(torch.load('checkpoint.pt'))
                    break

    def self_sl(self):
        x = self.u_set.x.detach()
        mask = mask_generator(x.shape, self.p_m)
        mask = mask.to(x.device)
        mask, x = pretext_generator(mask, x)

        u_loader = DataLoader(MaskDataset(x, mask), self.batch_size, shuffle=True)

        for e in range(self.self_epochs):
            with tqdm(u_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}") as pbar_epoch:
                for data in pbar_epoch:
                    x, mask = data
                    x = x.to(self.device)
                    mask = mask.to(self.device)
                    self.opt_self.zero_grad()

                    m_loss, f_loss = self.vime_self(x, mask)
                    loss = m_loss + self.alpha * f_loss
                    loss.backward()
                    self.opt_self.step()
                    pbar_epoch.set_description(f"epoch[{e + 1} / {self.self_epochs}]")
                    pbar_epoch.set_postfix({'loss': loss.item(),
                                            'mask loss': m_loss.item(),
                                            'feature loss': f_loss.item()})
            self.scheduler.step()

    def sl_only(self):
        idx = np.random.permutation(len(self.l_set))
        train_idx = idx[:int(len(idx) * 0.9)]
        valid_idx = idx[int(len(idx) * 0.9):]

        val_set = self.l_set[valid_idx]

        for i in range(self.semi_max_iter):
            b_idx = np.random.permutation(len(train_idx))[:self.batch_size].tolist()
            l_batch = self.l_set[train_idx[b_idx]]
            x_batch, y_batch = l_batch
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.opt_semi.zero_grad()
            pred = self.vime_semi(x_batch)
            loss = self.l_loss_fn(pred, y_batch)
            loss.backward()
            self.opt_semi.step()

            with torch.no_grad():
                x, y = val_set
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.vime_semi(x)
                val_loss = self.l_loss_fn(pred, y)

            if i % 100 == 0:
                print(f'Iteration: {i} / {self.semi_max_iter}, '
                      f'Current loss (val): {val_loss.item(): .4f}, '
                      f'Current loss (train): {loss.item(): .4f}, ')

            if i % int(self.l_samples / self.batch_size) == 0:
                self.early_stopping(val_loss, self.vime_semi)
                if self.early_stopping.early_stop:
                    print(f'early stopping {i} / {self.semi_max_iter}')
                    self.vime_semi.load_state_dict(torch.load('checkpoint.pt'))
                    break

    def test(self):
        test_loader = DataLoader(self.test_set, self.test_batch_size)
        results = []
        with tqdm(test_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}") as pbar_epoch:
            for data in pbar_epoch:
                with torch.no_grad():
                    x, y = data
                    x = x.to(self.device)

                    x = self.vime_self.encoder(x)
                    pred = self.vime_semi(x)
                    results.append(perf_metric('acc', y.cpu().numpy(), pred.cpu().numpy()))
        log.info(f'Performance: {100 * torch.tensor(results).mean()}')
