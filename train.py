import os
import sys
import time
import random
import argparse
import logging
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim

from models.cnexTIC import cnexTIC
from dataset import get_dataloader
from utils import init, Logger, setup_logger, load_checkpoint, save_checkpoint, AverageMeter
from losses.losses import Metrics, RateDistortionLoss


def parse_args(argv):
    parser = argparse.ArgumentParser(description='ConvNeXt-based Image Compression')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--name', help='result dir name', default='VCIP', type=str)
    parser.add_argument('--resume', help='snapshot path')
    parser.add_argument('--seed', help='seed number', default=None, type=int)
    args = parser.parse_args(argv)

    if not args.config:
        if args.resume:
            assert args.resume.startswith('./')
            dir_path = '/'.join(args.resume.split('/')[:-2])
            args.config = os.path.join(dir_path, 'config.yaml')
        else:
            args.config = './config.yaml'

    return args


def test(logger, test_dataloader, model, criterion, metric):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        logger.init()
        for i, x in enumerate(test_dataloader):
            x = x.to(device)
            out_net = model(x)
            out_criterion = criterion(out_net, x)
            bpp, psnr, ms_ssim = metric(out_net, x)

            logger.update_test(bpp, psnr, ms_ssim, out_criterion, model.aux_loss())

        logger.print_test()
        logger.write_test()

        loss.update(logger.loss.avg)
        bpp_loss.update(logger.bpp_loss.avg)
        mse_loss.update(logger.mse_loss.avg)
        logging.info(f'[ Test ] Total mean: {loss.avg:.4f}')
    logger.init()
    model.train()

    return loss.avg, bpp_loss.avg, mse_loss.avg


def train(args, config, base_dir, snapshot_dir, output_dir, log_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_id'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = RateDistortionLoss(lmbda=config['lmbda'], metric=config['metric'])
    metric = Metrics()
    train_dataloader, test_dataloader = get_dataloader(config)

    logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir)

    model = cnexTIC(config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    aux_optimizer = optim.Adam(model.aux_parameters(), lr=config['lr_aux'])
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 350], gamma=0.1)
    start_epoch = 0
    if args.resume:
        model = load_checkpoint(args.resume, model, optimizer, aux_optimizer)
        start_epoch = config['startepoch']

    # if config['set_lr']:
    #     lr_prior = optimizer.param_groups[0]['lr']
    #     for g in optimizer.param_groups:
    #         g['lr'] = float(config['set_lr'])
    #     print(f'[set lr] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')

    model.train()
    loss_best = 1e10
    # while logger.itr < config['max_itr']:
    for epoch in range(start_epoch,config['epochs']):
        # for epoch in range(281, config['epochs']):
        logging.info('======Current epoch %s ======' % epoch)
        for i, x in enumerate(train_dataloader):
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            x = x.to(device)
            out_net = model(x)
            out_criterion = criterion(out_net, x)

            out_criterion['loss'].backward()
            aux_loss = model.aux_loss()
            aux_loss.backward()

            # for stability
            if out_criterion['loss'].isnan().any() or out_criterion['loss'].isinf().any() or out_criterion[
                'loss'] > 100000:
                continue

            if config['clip_max_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_max_norm'])
            optimizer.step()
            aux_optimizer.step()  # update quantiles of entropy bottleneck modules

            # logging
            logger.update(i, out_criterion, aux_loss)
            if logger.itr % config['log_itr'] == 0:
                logger.print()
                logger.write()
                logger.init()

        # test and save model snapshot
        if epoch >= 0:
            model.update()
            loss, bpp_loss, mse_loss = test(logger, test_dataloader, model, criterion, metric)
            if loss < loss_best:
                logging.info('Best!')
                save_checkpoint(os.path.join(snapshot_dir, 'best.pt'), epoch, model, optimizer, aux_optimizer)
                loss_best = loss
            if epoch % config['snapshot_save_epoch'] == 0:
                save_checkpoint(os.path.join(snapshot_dir, f'{epoch:03}_{bpp_loss:.4f}_{mse_loss:.8f}.pt'),
                                epoch, model, optimizer, aux_optimizer)
        lr_scheduler.step()
        # # lr scheduling
        # if epoch % config['lr_shedule_step'] == 0:
        #     lr_prior = optimizer.param_groups[0]['lr']
        #     for g in optimizer.param_groups:
        #         g['lr'] *= config['lr_shedule_scale']
        #     print(f'[lr scheduling] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')


def main(argv):
    args = parse_args(argv)
    config, base_dir, snapshot_dir, output_dir, log_dir = init(args)
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        # torch.backends.cudnn.deterministic = True  # slow
        # torch.backends.cudnn.benchmark = False

    setup_logger(log_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    logging.info('[PID] %s' % os.getpid())
    logging.info('[config] %s' % args.config)
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k, v in config.items():
        if k in {'lr', 'set_lr', 'p'}:
            logging.info(f' *{k}: %s' % v)
        else:
            logging.info(f'  {k}: %s' % v)
    logging.info('=' * len(msg))

    train(args, config, base_dir, snapshot_dir, output_dir, log_dir)


if __name__ == '__main__':
    main(sys.argv[1:])