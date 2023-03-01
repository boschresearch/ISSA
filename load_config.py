from easydict import EasyDict as edict
import yaml
import copy
import math
from training.dataset import get_dataset_size



config = edict()

with open('./configs/mne_training.yml', 'r') as stream:
    opts = yaml.safe_load(stream)['encoder_training']
opts = edict(opts)


assert opts.noise_predict_from <= opts.noise_predict_until
log2_start = int(math.log2(opts.noise_predict_from))
log2_until = int(math.log2(opts.noise_predict_until))
all_noise_layers = 2 * ((log2_until - log2_start) + 1)
assert opts.num_masked_layer <= all_noise_layers

config.loss_kwargs = edict(class_name='training.loss.MixEncoderDiscriminatorLoss',
                      lambda_mse=opts.lambda_mse,
                      lambda_lpips=opts.lambda_lpips,
                      lambda_e_feat=opts.lambda_e_feat,
                      lambda_adv_loss=opts.lambda_adv_loss,
                      reconstruction_loss=opts.reconstruction_loss,
                      use_w=opts.use_w,
                      lambda_w=opts.lambda_w,
                      G_input_mode=opts.input_mode,
                      cooldown_w=opts.cooldown_w,
                      lambda_kl=opts.lambda_kl,
                      lambda_noise=opts.lambda_noise,
                      mask_ratio=opts.mask_ratio,
                      mask_size=opts.mask_size,
                      num_masked_layer=opts.num_masked_layer,
                      masked_noise_mode=opts.masked_noise_mode,
                      masked_noise_loss=opts.masked_noise_loss,
                      masked_lpips_loss=opts.masked_lpips_loss,
                      mask_height_divide=opts.mask_height_divide,
                      )
# normalize_layer_noise=opts.normalize_layer_noise,
# use_same_mask=opts.use_same_mask,
config.masked_lpips_loss = opts.masked_lpips_loss
config.loss_kwargs.blur_init_sigma = 10  # Blur the images seen by the discriminator.
config.loss_kwargs.blur_fade_kimg = opts.batch * 200 / 32  # Fade out the blur during the first N kimg.
config.loss_kwargs.gan_loss_mode = opts.loss_mode
config.loss_kwargs.r1_gamma = opts.gamma
config.loss_kwargs.enable_blur = opts.blur_enable
config.lr_schedule_kwargs = edict(lr=opts.lr, lr_decay_iter_start=opts.lr_decay_iter_start,
                             lr_decay_iter_end=opts.lr_decay_iter_end,
                             lr_decay=opts.lr_decay)

# Data loader configurations
config.data_loader_kwargs = edict(pin_memory=False, prefetch_factor=2)  # (pin_memory=True, prefetch_factor=2)

# Generator information
config.G_kwargs = edict(class_name=None, z_dim=opts.w_out_dim, w_dim=opts.w_out_dim, mapping_kwargs=edict())
config.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
config.G_kwargs.fused_modconv_default = 'inference_only'
config.G_kwargs.channel_base = 32768
config.G_kwargs.channel_max = 512
config.G_kwargs.mapping_kwargs.num_layers = 8
config.G_pkl = opts.pkl_dir
config.G_kwargs.input_mode = opts.input_mode  # 'const' or 'random'
config.cooldown_w = opts.cooldown_w
config.label_dim = opts.label_dim

# Encoder configs
config.encoder_kwargs = edict(class_name='training.encoder_model.FPNEncoder')
config.encoder_kwargs.style_layers = [opts.style_layers_coarse, opts.style_layers_medium, opts.style_layers_fine]
config.encoder_kwargs.fpn_feature_dim = opts.fpn_feature_dim
config.encoder_kwargs.out_dim = opts.w_out_dim
config.encoder_kwargs.noise_predict_from = opts.noise_predict_from
config.encoder_kwargs.noise_predict_until = opts.noise_predict_until
config.encoder_kwargs.input_mode = opts.input_mode

config.enc_D_kwargs = edict(class_name='training.discriminator_model.MultiscaleDiscriminator')

# Optimizer configs
if opts.optimizer == 'adam':
    config.enc_opt_kwargs = edict(class_name='torch.optim.Adam', lr=opts.lr)
else:
    config.enc_opt_kwargs = edict(class_name='training.ranger.Ranger', lr=opts.lr)

config.enc_D_opt_kwargs = edict(class_name='torch.optim.Adam', betas=[opts.beta_first, opts.beta_sec], eps=1e-8, lr=opts.dlr)

config.ema_enable = opts.ema_enable
config.ema_kimg =  opts.ema_kimg
config.ema_rampup = opts.ema_rampup
config.use_w = opts.use_w


dataset_name = opts.dataset_name


# Dataset Size
config.data_pth = opts.data
all_data = get_dataset_size(opts.data)
val_size = opts.val_size
train_size = all_data - val_size


# Training set
config.training_set_kwargs = edict(
    path=opts.data,
    resolution=opts.resolution,
    img_ratio=float(opts.img_ratio),
    cropping_mode=opts.cropping_dataset,
    use_labels = opts.cond,
    xflip = opts.mirror,
    max_size = train_size
)

# Validation set
config.val_set_kwargs = copy.deepcopy(config.training_set_kwargs)
config.val_set_kwargs.max_size = val_size
config.val_set_kwargs.inverse_order = True

# Fake image set
if opts.use_w:
    config.fake_set_kwargs = edict(
        path=opts.data_fake,
        resolution=opts.resolution,
        img_ratio=float(opts.img_ratio),
        cropping_mode=opts.cropping_dataset,
        use_w=opts.use_w,
        use_labels=opts.cond,
        xflip=opts.mirror
    )


config.encoder_kwargs.resolution = config.training_set_kwargs.resolution

# Hyperparameters & settings.
config.num_gpus = opts.gpus
config.batch_size = opts.batch
config.val_batch_size = opts.batch_val
config.batch_gpu = opts.subbatch or opts.batch // opts.gpus
config.total_kimg = opts.kimg
config.kimg_per_tick = opts.tick
config.image_snapshot_ticks = config.network_snapshot_ticks = opts.snap
config.random_seed = config.training_set_kwargs.random_seed = opts.seed
config.data_loader_kwargs.num_workers = opts.workers

# Sanity checks.
if config.batch_size % config.num_gpus != 0:
    raise ValueError('--batch must be a multiple of --gpus')
if config.batch_size % (config.num_gpus * config.batch_gpu) != 0:
    raise ValueError('--batch must be a multiple of --gpus times --batch-gpu')

# Resume.
if opts.resume is not None:
    config.resume_pkl = opts.resume
    # config.ada_kimg = 100  # Make ADA react faster at the beginning.

# Augmentation.
if opts.aug != 'noaug':
    config.augment_kwargs = edict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1,
                             aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
    if opts.aug == 'ada':
        config.ada_target = opts.target
        config.ada_max_p = opts.ada_max_p
        config.ada_interval = opts.ada_interval
        config.ada_kimg = opts.ada_kimg
        config.augment_p = 0.0
    if opts.aug == 'fixed':
        config.augment_p = opts.p
else:
    config.augment_kwargs = None

if opts.nobench:
    config.cudnn_benchmark = False
else:
    config.cudnn_benchmark = True

# Description string.
if opts.dataset_name is not None:
    dataset_name = opts.dataset_name

desc = f'encoder-{opts.generator:s}-{dataset_name:s}'

if opts.resume is not None:
    desc += '-resume'

config.desc = desc
config.outdir = opts.outdir


# Training setups
config.D_reg_interval = opts.D_reg_interval # freq. for lazy regularization
config.save_model_ticks = opts.save_model_ticks
config.image_snapshot_ticks = opts.image_snapshot_ticks