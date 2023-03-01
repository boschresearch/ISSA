import os
import json
import torch
from easydict import EasyDict as edict
from load_config import config
from datetime import datetime

from training.dataset import ImageDataset

import time
import copy
import pickle
import psutil
import PIL.Image
import numpy as np
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from training import lpips
import math
from importlib import import_module

# ToDo: To be deleted!
#from torch_utils import training_stats



#----------------------------------------------------------------------------


def create_class_by_name(*args, class_name=None, **kwargs):
    try:
        module_path, class_name = class_name.rsplit('.', 1)
        module = import_module(module_path)
        obj = getattr(module, class_name)
        assert callable(obj)
        return obj(*args, **kwargs)
    except (ImportError, AttributeError) as e:
        raise ImportError(class_name)


def initial_process(rank):
    if config.num_gpus > 0:
        os.environ['MASTER_ADDR'] = '127.0.0.4'
        os.environ['MASTER_PORT'] = '9904'
        torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                             rank=rank,world_size=config.num_gpus)
        print('---> Initialize torch.distributed!')

    # Init torch_utils.
    device = torch.device('cuda', rank) if config.num_gpus > 1 else None
    #training_stats.init_multiprocessing(rank=rank, sync_device=device)


#----------------------------------------------------------------------------

def create_output_dir(rank):
    desc = config.desc
    outdir = config.outdir
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    config.run_dir = os.path.join(outdir, f'{dt_string}-{desc}')

    # Create output directory.
    if rank ==0:
        print('Creating output directory...')
        print(config.run_dir)
        os.makedirs(config.run_dir)
        with open(os.path.join(config.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(config, f, indent=2)


def update_learning_rate(args, i, optimizer):
    if i < args.lr_decay_iter_start:
        pass
    elif i < args.lr_decay_iter_end:
        lr_max = args.lr
        lr_min = args.lr_decay
        t_max = args.lr_decay_iter_end - args.lr_decay_iter_start
        t_cur = i - args.lr_decay_iter_start

        optimizer.param_groups[0]['lr'] = lr_min + 0.5 * (lr_max - lr_min) * (
                    1 + math.cos(t_cur * 1.0 / t_max * math.pi))

def save_image_grid(img, fname, drange):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    p, _N, C, H, W = img.shape
    gw = 4
    gh = p*_N //gw
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(1, 3, 0, 4, 2)
    img = img.reshape([gw * H, gh * W, C])
    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
    return img.transpose(2,0,1)


class LoopedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

#----------------------------------------------------------------------------
def main(rank):
    create_output_dir(rank)
    initial_process(rank)
    num_gpus = config.num_gpus
    batch_size = config.batch_size
    random_seed = config.random_seed

    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(config.random_seed * num_gpus + rank)
    torch.manual_seed(config.random_seed * config.num_gpus + rank)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark  # Improves training speed.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')

    if config.use_w:
        divide_by = 2
    else:
        divide_by = 1

    #########################
    # Load datasets
    #########################
    # training set
    training_set = ImageDataset(**config.training_set_kwargs)
    training_set_sampler = LoopedSampler(dataset=training_set,
                                         rank=rank,
                                         num_replicas=num_gpus,
                                         seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                             batch_size=batch_size // num_gpus // divide_by,
                                                             **config.data_loader_kwargs))

    # validation set
    val_set = ImageDataset(**config.val_set_kwargs)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, shuffle=False, drop_last=True, sampler=val_sampler,
                                             batch_size=config.batch_gpu, **config.data_loader_kwargs)

    # fake image set
    if config.use_w:
        fake_set = ImageDataset(**config.fake_set_kwargs)
        fake_set_sampler = LoopedSampler(dataset=fake_set, rank=rank, num_replicas=num_gpus,seed=random_seed)
        fake_set_iterator = iter(torch.utils.data.DataLoader(dataset=fake_set, sampler=fake_set_sampler,
                                                             batch_size=batch_size // num_gpus // divide_by,
                                                             **config.data_loader_kwargs))
    if rank == 0:
        print()
        print('Num training images: ', len(training_set))
        print('Num validation images: ', len(val_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print('Aspect ratio: ', training_set.aspect_ratio)
        print()


    #########################
    # Load generator
    #########################
    common_kwargs = dict(c_dim=config.label_dim, img_resolution=training_set.resolution,
                         img_channels=training_set.num_channels, img_aspect_ratio=training_set.aspect_ratio)
    generator = create_class_by_name(**config.G_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)
    #dnnlib.util.construct_class_by_name
    # load pretrained Generator
    if rank == 0:
        print('Loading networks from "%s"...' % config.G_pkl)
        with open(config.G_pkl, "rb") as f:
            resume_data = misc.load_network_pkl(f)
        for name, module in [('G_ema', generator)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    #####################################
    # Construct Encoder & Discriminator
    #####################################
    if rank == 0:
        print('Constructing encoder...')
    common_kwargs = dict(input_dim=training_set.num_channels, n_latent=generator.mapping.num_ws)
    encoder = create_class_by_name(**config.encoder_kwargs , **common_kwargs).train().requires_grad_(
        False).to(device)
    encoder_ema = copy.deepcopy(encoder).eval()

    if rank == 0:
        print('Setting Discriminator...')
        D_channel = training_set.num_channels
        common_kwargs = dict(input_nc=D_channel, getIntermFeat=True)
        D_enc = create_class_by_name(**config.enc_D_kwargs, **common_kwargs).train().requires_grad_(
            False).to(device) # subclass of torch.nn.Module

    #############################
    # Load Perception network
    #############################
    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg',
        use_gpu=False, gpu_ids=[rank],
        masked_lpips_loss=config.masked_lpips_loss
    ).to(device)

    #############################
    # Configure Augmentation
    #############################
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    ada_max_p = torch.tensor(config.ada_max_p, device=device)

    # Not using augmentation pipeline in this code.
    # if (config.augment_kwargs is not None) and (config.augment_p > 0 or config.ada_target is not None):
    #     augment_pipe = create_class_by_name(**config.augment_kwargs).train().requires_grad_(
    #         False).to(device)
    #     augment_pipe.p.copy_(torch.as_tensor(config.augment_p))
    #     if config.ada_target is not None:
    #         ada_stats = training_stats.Collector(regex='Loss/E/lpips')


    #############################
    # Multi GPUs
    #############################
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [encoder, generator, percept, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)


    #############################
    # Setup training
    #############################
    if rank == 0:
        print('Setting up training ...')
    noise_mode = 'const'
    loss = create_class_by_name(
        device=device,
        E=encoder,D=D_enc, G=generator, percept=percept,
        augment_pipe=augment_pipe,
        noise_mode=noise_mode, **config.loss_kwargs
    )

    phases = [] # All modules(E & D) are in this list
    for name, module, opt_kwargs, reg_interval in [('E', encoder, config.enc_opt_kwargs, None),
                                                   ('D', D_enc, config.enc_D_opt_kwargs, config.D_reg_interval)]:
        if reg_interval is None:
            opt = create_class_by_name(params=module.parameters(), **opt_kwargs)
            phases += [edict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # lazy regularization
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = edict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = create_class_by_name(module.parameters(), **opt_kwargs)
            phases += [edict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [edict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]


    #############################
    # Initialize logsging
    #############################
    if rank == 0:
        print('Initializing logs...')
    #stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(config.run_dir, 'stats.jsonl'), 'wt')
        import torch.utils.tensorboard as tensorboard
        stats_tfevents = tensorboard.SummaryWriter(config.run_dir)
        print(f'Training for {config.total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0


    while True:
        # Fetch training data.
        if config.use_w:
            phase_syn_img, phase_gt_w = next(fake_set_iterator) # [bs,3,h,w], [bs, 512]
            repeat_w = int((np.log2(phase_syn_img.shape[-1]) - 1) * 2)
            phase_gt_w = phase_gt_w.unsqueeze(1).repeat(1,repeat_w,1).split(config.batch_gpu//divide_by)
            phase_syn_img = (phase_syn_img.to(device).to(torch.float32) / 127.5 - 1).split(config.batch_gpu//divide_by)
        else:
            phase_gt_w = None
            phase_syn_img = None
        phase_real_img = next(training_set_iterator) # [bs,3,h,w], [bs, 0]
        phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(config.batch_gpu//divide_by) # tuple

        if phase_gt_w is None:
            phase_gt_w = [None] * len(phase_real_img)
            phase_syn_img = [None] * len(phase_real_img)

        phase_new_img = [None] * len(phase_real_img)

        # -------------------> !!! ATTENTION !!! <-------------------#
        # Execute training phases.
        for phase in phases:
            if batch_idx % phase.interval != 0:
                continue
            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            for real_img, syn_img, gt_w, new_img in zip(phase_real_img,phase_syn_img, phase_gt_w, phase_new_img):
                if phase.name =='Eboth':
                    train_loss = loss.accumulate_gradients(phase=phase.name, real_img=real_img,
                                                           syn_img=syn_img,
                                                           gain=phase.interval, cur_nimg=cur_nimg,
                                                           new_img=new_img, gt_w=gt_w)
                else:
                    loss.accumulate_gradients(phase=phase.name, real_img=real_img,
                                              gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            params = [param for param in encoder.parameters() if param.grad is not None]
            if len(params) > 0:
                flat = torch.cat([param.grad.flatten() for param in params])
                if num_gpus > 1:
                    torch.distributed.all_reduce(flat)
                    flat /= num_gpus
                misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                grads = flat.split([param.numel() for param in params])
                for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
            phase.opt.step()
            if phase.name == 'Eboth':
                update_learning_rate(config.lr_schedule_kwargs, batch_idx, phase.opt)
                # learning rate
                #training_stats.report0('lr/e_lr', phase.opt.param_groups[0]['lr'])


        # Update E_ema.
        if config.ema_enable:
            ema_nimg = config.ema_kimg * 1000
            if config.ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * config.ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(encoder_ema.parameters(), encoder.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(encoder_ema.buffers(), encoder.buffers()):
                b_ema.copy_(b)
        else:
            encoder_ema = encoder

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % config.ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(config.ada_target - ada_stats['Loss/E/lpips']) \
                     * (batch_size * config.ada_interval) / (config.ada_kimg * 1000)
            temp_p = augment_pipe.p + adjust
            temp_p = min(temp_p, ada_max_p)
            augment_pipe.p.copy_((temp_p).max(misc.constant(0, device=device)))

        # Logging
        done = (cur_nimg >= config.total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + config.kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        # fields = []
        # fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        # fields += [
        #     f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        # fields += [
        #     f"cpumem {training_stats.report0('Resources/cpu_mem_G', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        # fields += [
        #     f"gpumem {training_stats.report0('Resources/peak_gpu_mem_G', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        # torch.cuda.reset_peak_memory_stats()
        # fields += [
        #     f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        # training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        # if rank == 0:
        #     print(' '.join(fields))
        if rank==0:
            print(f"Finish {cur_nimg} kimg.")

        # Save image snapshot.
        if (rank == 0) and (config.image_snapshot_ticks is not None) and (done or cur_tick % config.image_snapshot_ticks == 0):
            latent_w, input_noise, layer_noise = encoder(real_img)
            fake_img = generator.synthesis(latent_w, input_noise=input_noise, layer_noise=layer_noise,
                                            noise_predict_from=encoder.noise_predict_from,
                                            noise_predict_until=encoder.noise_predict_until,
                                            noise_mode='const')
            images = torch.stack([real_img.cpu(), fake_img.cpu()], dim=1).detach().numpy()
            save_image_grid(images,
                            fname=os.path.join(config.run_dir, f'rec-train-{cur_nimg // 1000:06d}.png'),
                            drange=[-1, 1],)
            del fake_img
            del real_img
            del images



        # Save model
        snapshot_pkl = None
        snapshot_data = None
        if (config.save_model_ticks is not None) and (done or cur_tick % config.save_model_ticks == 0):
            snapshot_data = dict(E=encoder_ema, augment_pipe=augment_pipe,
                                 training_set_kwargs=dict(config.training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        torch.distributed.barrier()
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value
            snapshot_pkl = os.path.join(config.run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
        del snapshot_data

        # Collect statistics.
        #stats_collector.update()
        #stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            #for name, value in stats_dict.items():
            #    stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        if done:
            break


#----------------------------------------------------------------------------

if __name__ == "__main__":
    if config.num_gpus <= 1:
        main(rank=0)
    else:
        torch.multiprocessing.spawn(main, nprocs=config.num_gpus, args=())


