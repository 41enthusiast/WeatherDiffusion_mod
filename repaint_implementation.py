import torch as th
import enum
import numpy as np
from collections import defaultdict
from torchvision.transforms.functional import crop
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image, to_tensor

class GaussianDiffusionRepaintWD_modded:
    def __init__(self,
                 betas,
                 corners,
                 x_grid_mask,
                 p_size,
                 device
                 ) -> None:
        
        self.device = device
        self.betas = betas
        self.corners = corners
        self.x_grid_mask = x_grid_mask
        self.p_size = p_size
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])

    def undo(self, img_after_model, t):
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = self.betas.index_select(0, t.long()).view(-1, 1, 1, 1).to(img_out.dtype)

        img_in_est = th.sqrt(1 - beta) * img_out + \
            th.sqrt(beta) * th.randn_like(img_out)

        return img_in_est
    
    def compute_alpha(self, t):
        beta = th.cat([th.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def p_sample(
            self,
            model,#unet noise estimator
            x,#x_{t-1}, current noisy image
            t,#startin at 0 for the first diffusion step
            gt,
            mask,
            pred_xstart=None,
        ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :return: a dict containing the following keys:
                    - 'sample': a random sample from the model.
                    - 'pred_xstart': a prediction of x_0.
        """
        # noise = th.randn_like(x)#gauss noise
        # STITCHING STEP
        if pred_xstart is not None:
            # print("Stitching step")
            gt_keep_mask = mask# later to uneti2i model where u get it
            alpha_cumprod = self.alpha_cumprod

            #weighting gt to same noise level as the noisy image
            gt_weight = th.sqrt(alpha_cumprod)
            gt_part = gt_weight * gt#*2 -1.0)

            noise_weight = th.sqrt((1 - alpha_cumprod))
            noise_part = noise_weight * th.randn_like(x)

            weighed_gt = gt_part + noise_part

            print("Stitching Range of the model inputs", x.min().item(), x.max().item(), "and gt", gt.min().item(), gt.max().item(), "and weighed_gt", weighed_gt.min().item(), weighed_gt.max().item())
            #the actual stitching of noisy image and noised ground truth image
            x = (
                gt_keep_mask * (weighed_gt)
                +
                (1 - gt_keep_mask) * (x)
            )
            to_pil_image(x.squeeze().cpu()).save("outputs/WeatherDiff64/stitched_image_t"+str(t.item())+".png")

        out = self.p_mean_variance(
            model,
            x,
            t,
            gt,
        )

        sample = out["mean"] 

        result = {"sample": sample,
                    "pred_xstart": out["pred_xstart"], 'gt': gt}

        return result

    def get_wd_processed_output(self, model, x, t, gt):
        manual_batching_size = 64
        patch_pbar = tqdm(range(0, len(self.corners), manual_batching_size), 
                        leave=False, 
                        desc=f"Processing Patches (t={t.item()})")
        et_output = th.zeros_like(x, device=x.device)#1x3xHxW
        # print("Range of the model inputs", x.min().item(), x.max().item(), "and gt", gt.min().item(), gt.max().item())
        for p in patch_pbar:
            current_batch_corners = self.corners[p:p+manual_batching_size]
            
            xt_patch = th.cat([crop(x, hi, wi, self.p_size, self.p_size) for (hi, wi) in current_batch_corners], dim=0)
            x_cond_patch = th.cat([crop(gt, hi, wi, self.p_size, self.p_size) for (hi, wi) in current_batch_corners], dim=0).to(x.device)
            
            outputs = model(th.cat([x_cond_patch, 
                                        xt_patch], dim=1), t)
            for idx, (hi, wi) in enumerate(self.corners[p:p+manual_batching_size]):
                et_output[0, :, hi:hi + self.p_size, wi:wi + self.p_size] += outputs[idx]  
        et = th.div(et_output, self.x_grid_mask)
        # print("Range of the model output before stitching", et.min().item(), et.max().item())
        return et
    
    def p_mean_variance(
            self,
            model,
            x, #[N x C x ...] tensor at time t.
            t, #1D timesteps
            gt,
        ):
            """
            Apply the model to get p(x_{t-1} | x_t) (noise eta), as well as a prediction of
            the initial x, x_0 (x_start).

            :param denoised_fn: if not None, a function which applies to the
                x_start prediction before it is used to sample. Applies before
                clip_denoised.
            :return: a dict with the following keys:
                    - 'mean': the model mean output.
                    - 'variance': the model variance output.
                    - 'log_variance': the log of 'variance'.
                    - 'pred_xstart': the prediction for x_0.
            """
            B, C = x.shape[:2]
            assert t.shape == (B,), t.shape

            # print('Getting noise estimate for ', t.item(), 'th timestep')
            model_output = self.get_wd_processed_output(model, x, t, gt)
            
            assert model_output.shape == (B, C, *x.shape[2:])

            #x0_t
            pred_xstart = (x - model_output * (1 - self.alpha_cumprod).sqrt()) / self.alpha_cumprod.sqrt()
            # eta = 0.0
            # c1 = eta * ((1 - self.alpha_cumprod / self.alpha_cumprod_prev) * (1 - self.alpha_cumprod_prev) / (1 - self.alpha_cumprod)).sqrt()
            c2 = (1 - self.alpha_cumprod_prev) #- c1 ** 2
            # print('Getting denoised sample')
            model_mean =  self.alpha_cumprod_prev.sqrt() * pred_xstart + c2.sqrt() * model_output #+ c1*th.randn_like(x) 
            
            assert (
                model_mean.shape == pred_xstart.shape == x.shape# == model_log_variance.shape
            )

            return {
                "mean": model_mean,
                "pred_xstart": pred_xstart,
            }
    
    def repaint_loop(
            self,
            model,#to get et after sampling all patches and averaging them over x_grid_mask
            gt,#masked image, since clean image is not available
            xt,
            mask,#gt mask to recover image regions from
            t_last,#seq, t, i
            t_cur,#seq_next, t_next, j
            jump_n_samples=5
        ):
            """
            Generate samples from the model and yield intermediate samples from
            each timestep of diffusion.

            Arguments are the same as p_sample_loop().
            Returns a generator over dicts, where each dict is the return value of
            p_sample().
            """

            print("Repainting step, t_cur:", t_cur.item(), "t_last:", t_last.item())
            self.gt_noises = None  # reset for next image. fixed noise consistency
            # logging_timesteps = []

            pred_xstart = None
            image_after_step = xt #th.randn(*gt.shape, device=gt.device)
            sample_idxs = defaultdict(lambda: 0)
            self.alpha_cumprod = self.compute_alpha(t_last.long())#to predict clean image
            self.alpha_cumprod_prev = self.compute_alpha(t_cur.long())#to determine how much clean signal to put back in, get the amount of noise to add in for the next step
            for _ in range(jump_n_samples):
                # if t_cur < t_last:  # reverse
                #DENOISE step
                # print('Denoising step')
                # logging_timesteps.append(t_last.item())
                # print("xt range", image_after_step.min().item(), image_after_step.max().item())
                with th.no_grad():
                    out = self.p_sample(
                        model,
                        image_after_step,
                        t_last,
                        gt,
                        mask,
                        pred_xstart=pred_xstart
                    )
                    #cleaner version of image is stored here
                    xt_next = out["sample"]
                    pred_xstart = out["pred_xstart"]
                    sample_idxs[t_cur] += 1
                    
                # else:
                # JUMPBACK step- to harmonize the stitched image at the previous step
                # print('Jumpback step')
                # logging_timesteps.append(t_cur.item())
                t_shift = abs(t_cur - t_last)
                image_after_step = self.undo(
                    xt_next,
                    t=t_cur+t_shift)
                pred_xstart = pred_xstart
                
            return xt_next


