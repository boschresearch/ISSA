import numpy as np
import torch
#from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
import torch.nn.functional as F
from training.masking import mask_images


#----------------------------------------------------------------------------
class GANLoss(torch.nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor,device=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.device = device


    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)


    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0).to(device=self.device)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)


    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            pass


    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

#----------------------------------------------------------------------------


class MixEncoderDiscriminatorLoss():
    def __init__(self, device, E, G, D, percept, noise_mode='const', augment_pipe=None,
                 gan_loss_mode='hinge', reconstruction_loss='L2',
                 lambda_lpips=10.0, lambda_mse=1.0,
                 lambda_e_feat=2.0, lambda_adv_loss=1.0,
                 use_w=False, lambda_w=10.0,
                 lambda_kl=0.0, lambda_noise=1.0,
                 cooldown_w=False, G_input_mode='random',
                 r1_gamma=10, enable_blur=False,
                 blur_init_sigma=0, blur_fade_kimg=0,
                 mask_ratio=0.0, mask_size=2, num_masked_layer=2,
                 masked_noise_mode='normal',
                 masked_noise_loss=False,
                 masked_lpips_loss=False,
                 start_ada_from=10000,
                 use_same_mask=False, mask_height_divide=16,
                 ):

        super().__init__()
        self.device             = device
        self.E                  = E
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.start_ada_from     = start_ada_from
        self.r1_gamma           = r1_gamma
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.percept            = percept
        self.gan_loss           = GANLoss(gan_mode=gan_loss_mode, tensor=torch.FloatTensor, device=device).to(device)
        self.reconstruction_loss= reconstruction_loss
        self.lambda_lpips       = lambda_lpips
        self.lambda_mse         = lambda_mse
        self.lambda_e_feat      = lambda_e_feat
        self.lambda_adv_loss    = lambda_adv_loss
        self.use_w              = use_w
        self.lambda_w           = lambda_w
        self.cooldown_w         = cooldown_w
        self.lambda_kl          = lambda_kl
        self.lambda_noise       = lambda_noise

        # About Masking
        self.mask_ratio         = mask_ratio
        self.use_same_mask      = use_same_mask
        self.mask_height_divide = mask_height_divide
        self.given_mask         = None
        self.mask_size = [mask_size, mask_size]

        self.noise_mode         = noise_mode
        self.enable_blur        = enable_blur
        self.device             = device
        self.G_input_mode       = G_input_mode
        self.input_p            = torch.distributions.Normal(torch.tensor(0).to(device=device),
                                                             torch.tensor(1).to(device=device))
        self.noise_predict_from = E.noise_predict_from
        self.noise_predict_until= E.noise_predict_until
        self.num_masked_layer   = num_masked_layer
        self.masked_noise_mode  = masked_noise_mode
        self.masked_noise_loss  = masked_noise_loss
        self.masked_lpips_loss  = masked_lpips_loss

    def run_G(self, real_img):
        latent_w, input_noise, layer_noise = self.E(real_img)
        layer_noise_copy = layer_noise.copy()

        N,c,h,w = real_img.shape
        all_masks = torch.ones(N, 1, h, w)
        if self.mask_ratio > 0 :
            img_ratio = layer_noise[0].shape[-1]/layer_noise[0].shape[-2]
            masked_layer_noise = layer_noise[:-1 * self.num_masked_layer]
            if self.masked_noise_loss:
                all_masks = torch.ones(N, 1,h, w)
                for i in range(-1  * self.num_masked_layer, 0):
                    if self.use_same_mask :
                        mask_size = layer_noise[i].shape[-2] // self.mask_height_divide
                        mask_size = [mask_size, mask_size]
                        if i != (-1  * self.num_masked_layer):
                            masked_noise, masks, self.given_mask = mask_images(layer_noise[i], mask_size, self.mask_ratio,
                                                                               img_ratio=img_ratio,
                                                                               masked_noise_mode=self.masked_noise_mode,
                                                                               given_mask=self.given_mask)
                        else:
                            masked_noise, masks, self.given_mask = mask_images(layer_noise[i], mask_size,
                                                                               self.mask_ratio,
                                                                               img_ratio=img_ratio,
                                                                               masked_noise_mode=self.masked_noise_mode,)

                    else:
                        masked_noise, masks,self.given_mask = mask_images(layer_noise[i], self.mask_size,
                                                                          self.mask_ratio,
                                                                          img_ratio=img_ratio,
                                                                          masked_noise_mode=self.masked_noise_mode)


                    masked_layer_noise.append(masked_noise)
                    masks = torch.logical_not(masks).to(torch.float32) # after not: 1 is keeping, 0 is removing
                    up_m = torch.nn.Upsample(scale_factor=real_img.shape[-1] // masks.shape[-1], mode='nearest')
                    masks = up_m(masks)
                    all_masks = torch.logical_and(all_masks, masks[:,0:1,:,:].to(all_masks.device))
            else:
                for i in range(-1  * self.num_masked_layer, 0):
                    masked_noise, masks, _ = mask_images(layer_noise[i], self.mask_size, self.mask_ratio,
                                                         img_ratio=img_ratio,masked_noise_mode=self.masked_noise_mode)
                    masked_layer_noise.append(masked_noise)
        else:
            masked_layer_noise = layer_noise # [bs, 512,32,64]

        fake_img = self.G.synthesis(latent_w, input_noise=input_noise, layer_noise=masked_layer_noise,
                                    noise_predict_from=self.noise_predict_from,
                                    noise_predict_until=self.noise_predict_until,
                                    noise_mode=self.noise_mode)
        return fake_img, latent_w, input_noise, layer_noise_copy, all_masks


    def blur_image(self, img, blur_sigma=0):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
            img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        return img


    def prep_d_input(self, img, img2):
        d_in = torch.cat([img, img2], dim=1)
        return d_in


    def prep_d_output(self, pred, use_feat=False):
        if use_feat:
            return pred
        else:
            for i in range(len(pred)):
                for j in range(len(pred[i]) - 1):
                    pred[i][j] = pred[i][j].detach()
            return pred


    def d_r1_loss(self, real_pred, real_img):
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True, only_inputs=True,
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty


    def warmup(self, start, end, max_steps, current_step):
        if current_step > max_steps:
            return end
        return (end - start) * (current_step / max_steps) + start


    def cooldown(self, lambda_init, current_step, start=200000, end=500000):
        if current_step < start:
            return lambda_init
        if current_step > end:
            return 0.0
        return lambda_init * (end-current_step) / (end-start)


    def denormalize_img(self, x):
        return (x + 1.0) / 2.0 # [-1, 1] => [0, 1]


    def multi_accumulate(self,phase, real_img, gain, cur_nimg, new_img=None,gt_w=None,syn_img=None):
        assert phase in ['Eboth', 'Dmain', 'Dreg', 'Dboth']
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3),
                         0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        if self.augment_pipe is not None and cur_nimg > self.start_ada_from:
            real_img = self.augment_pipe(real_img)

        # Eboth: train encoder
        if phase in ['Eboth']:
            real_num = real_img.shape[0]
            if syn_img is not None:
                imgs = torch.cat([real_img, syn_img], dim=0)
            else:
                imgs = real_img

            fake_img, latent_w, input_noise,layer_noise,all_masks = self.run_G(imgs)
            # all_masks: [bs,1,256,512]


            fake_g_pred = self.D(fake_img[:real_num])
            real_g_pred = self.D(real_img)


            fake_g_pred = self.prep_d_output(fake_g_pred, use_feat=True)
            real_g_pred = self.prep_d_output(real_g_pred, use_feat=False)
            e_adv_loss = self.gan_loss(fake_g_pred, True, for_discriminator=False).mean()

            # e feat loss
            e_feat_loss = 0.0
            feat_weights = 1.0
            D_weights = 1.0 / 3.0
            for D_i in range(len(fake_g_pred)):
                for D_j in range(len(fake_g_pred[D_i]) - 1):
                    e_feat_loss += D_weights * feat_weights * \
                                    F.l1_loss(fake_g_pred[D_i][D_j], real_g_pred[D_i][D_j].detach())
            #training_stats.report('Loss/fake/adv_loss', e_adv_loss)
            #training_stats.report('Loss/fake/feat_loss', e_feat_loss)

            # Pixel-level reconstuction loss
            if self.reconstruction_loss == 'L2':
                e_mse_loss = F.mse_loss(fake_img * all_masks.to(fake_img.device), imgs * all_masks.to(imgs.device))
            else:
                e_mse_loss = F.l1_loss(fake_img * all_masks.to(fake_img.device), imgs * all_masks.to(imgs.device))

            # Lpips loss
            if self.masked_lpips_loss:
                e_lpips_loss = self.percept(fake_img, imgs, mask=all_masks).mean()
            else:
                e_lpips_loss = self.percept(fake_img, imgs).mean()


            # Input noise loss
            if self.E.input_mode == 'const':
                kl_loss = 0.0
            else:
                mu = torch.mean(input_noise, dim=(-1), keepdims=False)
                mu = torch.nan_to_num(mu)
                if torch.isnan(mu).any():
                    print('Nan in mu')
                var = torch.var(input_noise, dim=(-1))
                q = torch.distributions.Normal(mu,var)
                kl = torch.distributions.kl_divergence(q, self.input_p).mean()
                #training_stats.report('Loss/E/input_KL', kl)
                kl_loss = kl * self.lambda_kl

            # Layer noise loss
            noise_sum = 0.0
            noise_shape_count = 0.0
            for noise_ in layer_noise:
                noise_ = noise_.view(-1)
                noise_shape_count += noise_.shape[0]
                noise_sum += torch.abs(noise_).sum()
            layer_noise_loss = noise_sum / noise_shape_count
            #training_stats.report('Loss/E/layer_nosie', layer_noise_loss)
            layer_noise_loss = self.lambda_noise * layer_noise_loss

            gt_w_loss = 0.0
            if self.use_w and gt_w is not None:
                #if gt_w is None:
                #    raise ValueError('Cannot find ground truth latent w!')
                _N = latent_w.shape[1]
                gt_w_loss = F.mse_loss(latent_w[real_num:], gt_w[:,-1*_N:,:].to(self.device)).mean()
                #training_stats.report('Loss/E/gt_w', gt_w_loss)
                if self.cooldown_w:
                    lambda_cur_w = self.cooldown(self.lambda_w,cur_nimg)
                else:
                    lambda_cur_w = self.lambda_w
                gt_w_loss = gt_w_loss * lambda_cur_w


            e_unlabel_loss = e_mse_loss * self.lambda_mse + e_lpips_loss * self.lambda_lpips \
                             + e_adv_loss * self.lambda_adv_loss + e_feat_loss * self.lambda_e_feat \
                             + gt_w_loss + kl_loss + layer_noise_loss

            # training_stats.report('Loss/E/mse', e_mse_loss)
            # training_stats.report('Loss/E/lpips', e_lpips_loss)
            # training_stats.report('Loss/E/all', e_unlabel_loss)
            e_unlabel_loss.backward()
            return e_unlabel_loss

        # Dmain: Minimize hinge loss
        if phase in ['Dmain', 'Dboth']:

            fake_img, _, _, _,all_masks = self.run_G(real_img)

            fake_pred = self.D(fake_img)


            fake_pred = self.prep_d_output(fake_pred, use_feat=False)

            d_fake_score = (fake_pred[0][-1].mean() + fake_pred[1][-1].mean() +
                            fake_pred[2][-1].mean()) / 3.0
            d_fake = self.gan_loss(fake_pred, False, for_discriminator=True).mean()
            #training_stats.report('Loss/scores/fake', d_fake_score)
            d_fake.mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])

            if self.enable_blur:
                blur_real_img = self.blur_image(real_img_tmp, blur_sigma=blur_sigma)
                real_img_tmp = blur_real_img
            else:
                real_img_tmp = real_img_tmp

            real_pred = self.D(real_img_tmp)
            real_pred = self.prep_d_output(real_pred, use_feat=False)
            d_real_score = (real_pred[0][-1].mean() + real_pred[1][-1].mean() +
                            real_pred[2][-1].mean()) / 3.0
            #training_stats.report('Loss/scores/real', d_real_score)

            d_real = 0
            if phase in ['Dmain', 'Dboth']:
                d_real = self.gan_loss(real_pred, True, for_discriminator=True).mean()
                #training_stats.report('Loss/D/loss', d_real + d_fake)

            loss_Dr1 = 0
            if phase in ['Dreg', 'Dboth']:
                real_pred = real_pred[0][-1].mean() + real_pred[1][-1].mean() + real_pred[2][-1].mean()
                r1_penalty = self.d_r1_loss(real_pred, real_img_tmp)
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                #training_stats.report('Loss/r1_penalty', r1_penalty)
                #training_stats.report('Loss/D/reg', loss_Dr1)

            (loss_Dr1 + d_real).mean().mul(gain).backward()


    def accumulate_gradients(self, phase, real_img, gain, cur_nimg, new_img=None, gt_w=None, syn_img=None):
        return self.multi_accumulate(phase, real_img, gain, cur_nimg, new_img, gt_w, syn_img)
