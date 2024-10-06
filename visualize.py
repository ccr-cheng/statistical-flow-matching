import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from utils import recursive_to_device

_VIS_DICT = {}


def register_vis(name):
    def decorator(cls):
        _VIS_DICT[name] = cls
        return cls

    return decorator


def get_vis(cfg, writer, device):
    v_cfg = cfg.copy()
    v_type = v_cfg.pop('type')
    return _VIS_DICT[v_type](writer, device, **v_cfg)


@register_vis('bmnist')
class MNISTVisualizer:
    def __init__(self, writer, device, n_sample, n_step):
        self.writer = writer
        self.device = device
        self.n_sample = n_sample
        self.n_step = n_step

    def __call__(self, model, method, global_step):
        traj = model.sample(method, self.n_sample, self.n_step, self.device)
        img = (traj[..., 0] > traj[..., 1]).float().view(-1, 28, 28)
        img = make_grid(img.unsqueeze(1), nrow=10, padding=2, pad_value=0)
        self.writer.add_image('sample', img, global_step)
        return traj


@register_vis('promoter')
class PromoterVisualizer:
    def __init__(self, writer, device, n_step, sei_model_path, sei_feat_path):
        import pandas as pd
        from evaluation import Sei, NonStrandSpecific

        self.writer = writer
        self.device = device
        self.n_step = n_step
        self.sei_model_path = sei_model_path
        self.sei_feat_path = sei_feat_path

        self.sei = NonStrandSpecific(Sei(sequence_length=4096, n_genomic_features=21907).to(device))
        state_dict = torch.load(sei_model_path, map_location='cpu')['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.sei.load_state_dict(state_dict)
        self.sei.eval()

        df = pd.read_csv(sei_feat_path, sep='|', header=None)
        self.sei_feat_idx = df[1].str.strip().values == 'H3K4me3'

    def __call__(self, model, dataloader, method, global_step, max_batch=None):
        gt_values = []
        pred_values = []
        traj = []
        if max_batch is None:
            max_batch = len(dataloader)

        for idx, (x, *cond_args) in enumerate(dataloader):
            if idx >= max_batch:
                break
            x = x.to(self.device)
            cond_args = recursive_to_device(cond_args, self.device)
            tj = model.sample(method, x.size(0), self.n_step, self.device, *cond_args).detach()
            pred_onehot = F.one_hot(tj.argmax(-1), num_classes=tj.size(-1)).float()
            traj.append(tj)

            ones = torch.ones((x.shape[0], 4, 1536), dtype=torch.float, device=self.device) * 0.25
            gt = self.sei(torch.cat([ones, x.transpose(1, 2), ones], 2)).detach().cpu().numpy()
            gt_values.append(gt[:, self.sei_feat_idx].mean(axis=1))
            pred = self.sei(torch.cat([ones, pred_onehot.transpose(1, 2), ones], 2)).detach().cpu().numpy()
            pred_values.append(pred[:, self.sei_feat_idx].mean(axis=1))
        traj = torch.cat(traj, 0)
        gt_values = np.concatenate(gt_values, axis=0)
        pred_values = np.concatenate(pred_values, axis=0)

        loss = ((gt_values - pred_values) ** 2).mean()
        self.writer.add_scalar('sample/mse', loss, global_step)
        print(f'Step {global_step} generation MSE {loss:.6f}')
        return traj


@register_vis('text8')
class Text8Visualizer:
    def __init__(self, writer, device, n_sample, n_step):
        self.writer = writer
        self.device = device
        self.n_sample = n_sample
        self.n_step = n_step
        self.TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz")

    def char_ids_to_str(self, char_ids) -> str:
        """Decode a 1D sequence of character IDs to a string."""
        return ''.join([self.TEXT8_CHARS[i] for i in char_ids])

    def batch_to_str(self, text_batch) -> list[str]:
        """Decode a batch of character IDs to a list of strings."""
        return [self.char_ids_to_str(row_char_ids) for row_char_ids in text_batch]

    def __call__(self, model, method, global_step):
        traj = model.sample(method, self.n_sample, self.n_step, self.device)
        txt = self.batch_to_str(traj.argmax(-1).tolist())
        for i, t in enumerate(txt):
            self.writer.add_text(f'sample{i}', t, global_step)
        return traj
