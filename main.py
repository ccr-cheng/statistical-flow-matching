import argparse
import os
import time

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import get_flow_model
from datasets import get_dataset
from utils import seed_all, load_config, get_optimizer, get_scheduler, count_parameters, recursive_to_device
from visualize import get_vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'inf'], default='train')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--savename', type=str, default='test')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)
    print(config)
    logdir = os.path.join(args.logdir, args.savename)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    visualizer = get_vis(config.visualizer, writer, args.device)

    # Data
    print('Loading datasets...')
    train_set, valid_set, test_set = get_dataset(config.datasets)

    # Dataloader
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_set, batch_size=config.train.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=config.train.batch_size, shuffle=False, num_workers=8)

    # Model
    print('Building model...')
    model = get_flow_model(config.model, config.encoder).to(args.device)
    print(f'Number of parameters: {count_parameters(model)}')

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()

    # Resume
    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
    global_step = 0


    def train():
        global global_step

        epoch = 0
        while True:
            model.train()
            epoch_losses = []
            for x, *cond_args in train_loader:
                # Training
                x = x.to(args.device)
                cond_args = recursive_to_device(cond_args, args.device)
                loss = model.get_loss(x, *cond_args)
                epoch_losses.append(loss.item())
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/grad', grad_norm.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                if global_step % config.train.log_freq == 0:
                    print(f'Epoch {epoch} Step {global_step} train loss {loss.item():.6f}')
                global_step += 1

                # Validation
                if global_step % config.train.val_freq == 0:
                    avg_val_loss = validate(valid_loader)
                    sample('euler', 'valid', config.get('sample_max_batch', None))
                    if config.train.scheduler.type == 'plateau':
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    model.train()
                    torch.save({
                        'model': model.state_dict(),
                        'step': global_step,
                    }, os.path.join(logdir, 'latest.pt'))
                    if global_step % config.train.save_freq == 0:
                        ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'avg_val_loss': avg_val_loss,
                        }, ckpt_path)
                if global_step >= config.train.max_iter:
                    return

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch} train loss {epoch_loss:.6f}')
            epoch += 1


    def validate(dataloader, split='valid'):
        with torch.no_grad():
            model.eval()

            val_losses = []
            total = config.get('valid_max_batch', None)
            if total is None:
                total = len(dataloader)
            for i, (x, *cond_args) in tqdm(enumerate(dataloader), total=total):
                if i >= total:
                    break
                x = x.to(args.device)
                cond_args = recursive_to_device(cond_args, args.device)
                loss = model.get_loss(x, *cond_args)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        writer.add_scalar(f'{split}/loss', val_loss, global_step)
        print(f'Step {global_step} {split} loss {val_loss:.6f}')
        return val_loss


    def sample(method='euler', split='valid', max_batch=None):
        with torch.no_grad():
            model.eval()
            if not config.conditioned:
                traj = visualizer(model, method, global_step)
            else:
                dataloader = valid_loader if split == 'valid' else test_loader
                traj = visualizer(model, dataloader, method, global_step, max_batch=max_batch)
        return traj


    try:
        if args.mode == 'train':
            train()
            print('Training finished!')
        if args.mode == 'inf' and args.resume is None:
            print('[WARNING]: inference mode without loading a pretrained model')

        sample('ode', 'valid', None)
        print('Sampling finished!')
        time.sleep(3)  # Wait for the last tensorboard logs to be written
    except KeyboardInterrupt:
        print('Terminating...')
