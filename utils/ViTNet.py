import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.loss import loss_function_dict
import utils.plot as utils_plot
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt # Add this line
import matplotlib.colors as pltclr # Add this line


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, C, H, W -> B, D, H/p, W/p -> B, D, N -> B, N, D
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTNet(nn.Module):
    verbose = False

    def __init__(
        self,
        input_type,
        input_channels: int,
        output_channels: int,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_rate: float = 0.0,
        # Decoder specific parameters
        decoder_channels: list = [256, 128, 64, 32],
        # Optional CNN pre/post-processing
        pre_cnn_channels: list = [],
        post_cnn_channels: list = [],
        optimizer_hparams=None,
        loss_hparams=None,
        logger_params=None,
        name=None,
        model_idx=None,
        device=None, # Add device parameter
        **kwargs,  # To capture other hparams from model_kwargs
    ):
        super().__init__()

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.name = name
        self.input_type = input_type
        self.loss_hparams = loss_hparams

        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.output_channels = output_channels

        # 1. Optional Pre-processing CNN
        self.pre_cnn = nn.Sequential()
        if pre_cnn_channels:
            in_ch = input_channels
            for out_ch in pre_cnn_channels:
                self.pre_cnn.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
                self.pre_cnn.append(nn.ReLU())
                in_ch = out_ch
            input_channels = pre_cnn_channels[-1]

        # 2. Patch Embedding
        self.patch_embed = PatchEmbedding(input_channels, patch_size, embed_dim, img_size)

        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim, device=self.device)) # Initialize on device
        self.pos_drop = nn.Dropout(p=dropout_rate)
        # 3. Transformer Encoder
        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, mlp_ratio, drop=dropout_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Decoder
        self.decoder_channels = decoder_channels
        self.decoder = nn.Sequential()
        in_decoder_ch = embed_dim
        # Initial convolution to reshape from transformer output to image-like features
        self.decoder.append(nn.Conv2d(in_decoder_ch, decoder_channels[0], kernel_size=1))

        for i in range(len(decoder_channels) - 1):
            self.decoder.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.decoder.append(nn.Conv2d(decoder_channels[i], decoder_channels[i+1], kernel_size=3, padding=1))
            self.decoder.append(nn.ReLU())

        # Final layer to match output channels and original image size
        self.final_conv = nn.Conv2d(decoder_channels[-1], output_channels, kernel_size=1)

        # 5. Optional Post-processing CNN
        self.post_cnn = nn.Sequential()
        if post_cnn_channels:
            in_ch = output_channels
            for out_ch in post_cnn_channels:
                self.post_cnn.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
                self.post_cnn.append(nn.ReLU())
                in_ch = out_ch
            self.final_conv = nn.Conv2d(post_cnn_channels[-1], output_channels, kernel_size=1)

        # Optimizer and Loss (copied from UNet)
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                             lr=optimizer_hparams['LR'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, optimizer_hparams['schedule_rate'])

        self.logger_params = logger_params
        self.logdir = logger_params['log_dir'] + '_%u' % model_idx
        self.logger = SummaryWriter(self.logdir)
        self.index = model_idx
        self.logger.add_text('Name', self.name, global_step=0)

        self.loss_function = loss_function_dict[loss_hparams['loss_type']](**loss_hparams)

        self.track_activations = False
        self.angmag = True if 'am' in loss_hparams['loss_type'] else False

        self.running_train_loss = {}
        self.n_training_batches = 0
        self.running_val_loss = {}
        self.n_validation_batches = 0
        self.sample_chosen_for_callback = False
        self.first_validation_sample = None

    def select_inputs(self, str_, dict_):
        keys = str_.split(",")
        inputs = []
        for key in keys:
            inputs.append(dict_[key])
        inputs = torch.cat(inputs, axis=1)
        if self.verbose:
            mask_shape_str = f", dict entry shape\t{dict_["mask"].shape}" if "mask" in dict_ else ""
            print(f" Models: Inputs shape: \t {inputs.shape}{mask_shape_str}")
        return inputs

    def reg_scheduler(self, schedule, x, e):
        if schedule['type'] == 'sigmoid':
            e_crit = schedule['e_crit']
            width = schedule['width']
            return x / (1 + np.exp(-(e - e_crit) / width))
        if schedule['type'] == 'linear':
            e_crit = schedule['e_crit']
            width = schedule['width']
            return x * np.maximum((e - e_crit) / width, 0)
        if schedule['type'] == 'none':
            return x

    def strainenergy_loss(self, Fpred, Fexp, Uexp, mask):
        if self.angmag:
            xt = Fexp[:, 0] * torch.cos(Fexp[:, 1])
            yt = Fexp[:, 0] * torch.sin(Fexp[:, 1])
            xp = Fpred[:, 0] * torch.cos(Fpred[:, 1])
            yp = Fpred[:, 0] * torch.sin(Fpred[:, 1])
            Fexp = torch.cat([xt.unsqueeze(1), yt.unsqueeze(1)], axis=1)
            Fpred = torch.cat([xp.unsqueeze(1), yp.unsqueeze(1)], axis=1)
        W_pred = torch.sum(Fpred * Uexp, axis=1, keepdim=True)
        W_exp = torch.sum(Fexp * Uexp, axis=1, keepdim=True)
        W_pred = W_pred[mask != 0].mean()
        W_exp = W_exp[mask != 0].mean()
        return (W_pred - W_exp).pow(2)

    def forward(self, x):
        # Pre-processing CNN
        if self.pre_cnn:
            x = self.pre_cnn(x)

        # Patch Embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer Encoder
        x = self.blocks(x)
        x = self.norm(x)

        # Reshape for Decoder
        B, N, C = x.shape
        H_feat = W_feat = int(N**0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)

        # Decoder
        # Initial conv (if any, already handled by self.decoder[0])
        x = self.decoder[0](x) 

        # Iterate through decoder layers, handling Upsample and Conv2d
        # Assuming decoder_channels define the output channels of each Conv2d after Upsample
        # and that Upsample layers are followed by Conv2d layers.
        current_h, current_w = H_feat, W_feat # Start with feature map size
        for i in range(1, len(self.decoder)):
            layer = self.decoder[i]
            if isinstance(layer, nn.Upsample):
                # Calculate target size based on original img_size and current scale
                # This assumes a fixed upsampling strategy to reach img_size
                # A more robust approach might involve calculating the required output size
                # for each upsample step to precisely hit img_size at the end.
                scale_factor = layer.scale_factor
                current_h = int(current_h * scale_factor)
                current_w = int(current_w * scale_factor)
                # Ensure we don\'t exceed img_size during intermediate upsampling
                current_h = min(current_h, self.img_size)
                current_w = min(current_w, self.img_size)
                x = F.interpolate(x, size=(current_h, current_w), mode=\'bilinear\', align_corners=False)
            else:
                x = layer(x)

        # Ensure output matches original image size after all decoder operations
        # This final interpolation acts as a safeguard.
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode=\'bilinear\', align_corners=False)

        x = self.final_conv(x)

        # Post-processing CNN
        if self.post_cnn:
            x = self.post_cnn(x)

        return x

    def training_step(self, batch, epoch=None):
        self.train()
        self.optimizer.zero_grad()

        prediction = self(self.select_inputs(self.input_type, batch))

        # Base loss calculation
        loss_dict_raw = self.loss_function(prediction, batch["output"], expweight=expweight)
        base_loss = loss_dict_raw["base_loss"] # Keep base_loss for backward pass

        # Strain energy loss calculation
        strainenergy_loss_raw = self.strainenergy_loss(prediction, batch["output"], batch["displacements"], batch["mask"].bool())
        
        # Total loss for backward pass
        loss = base_loss \
                + self.reg_scheduler(self.loss_hparams.get("reg_schedule"),
                                     self.loss_hparams.get("strainenergy_regularization"),
                                     epoch) * strainenergy_loss_raw # Use raw strainenergy_loss

        if torch.isnan(loss):
            print("LOSS IS NAN")
            print(loss_dict_raw, {"strainenergy_loss": strainenergy_loss_raw})

        loss.backward()
        self.optimizer.step()

        # Detach for logging AFTER backward pass
        loss_dict_logged = {key: item.detach() for key, item in loss_dict_raw.items()}
        loss_dict_logged = {**loss_dict_logged, 
                            "strainenergy_loss": strainenergy_loss_raw.sqrt().detach(), # Detach here
                            "exp_schedule": torch.tensor(expweight).detach(), # Ensure expweight is a tensor if needed for logging
                            "reg_schedule": torch.tensor(self.reg_scheduler(self.loss_hparams.get("reg_schedule"), 1., epoch)).detach()}

        if not self.running_train_loss:
            self.running_train_loss = loss_dict
        else:
            self.running_train_loss = {key: item + loss_dict[key] for key, item in self.running_train_loss.items()}
        self.n_training_batches += 1

        return

    def validation_step(self, batch, epoch=None):
        self.eval()

        with torch.no_grad():
            prediction = self(self.select_inputs(self.input_type, batch))

            expweight = self.loss_hparams.get("exp_weight") * self.reg_scheduler(self.loss_hparams.get("exp_schedule"), 1., epoch)
            loss_dict_raw = self.loss_function(prediction, batch["output"], expweight=expweight)

            strainenergy_loss_raw = self.strainenergy_loss(prediction, batch["output"], batch["displacements"], batch["mask"].bool())
            loss = loss_dict_raw["base_loss"] \
                + self.reg_scheduler(self.loss_hparams.get("reg_schedule"), self.loss_hparams.get("strainenergy_regularization"), epoch) * strainenergy_loss_raw

            # Detach for logging
            loss_dict_logged = {key: item.detach() for key, item in loss_dict_raw.items()}
            loss_dict_logged = {**loss_dict_logged, 
                                "strainenergy_loss": strainenergy_loss_raw.sqrt().detach(),
                                "exp_schedule": torch.tensor(expweight).detach(),
                                "reg_schedule": torch.tensor(self.reg_scheduler(self.loss_hparams.get("reg_schedule"), 1., epoch)).detach()}

            if not self.running_val_loss:
                self.running_val_loss = loss_dict
            else:
                self.running_val_loss = {key: item + loss_dict[key] for key, item in self.running_val_loss.items()}

            if not self.sample_chosen_for_callback:
                self.first_validation_sample = {**batch, 'prediction': prediction.detach()}
                if prediction.shape[0] >= 2:
                    self.sample_chosen_for_callback = True

        self.n_validation_batches += 1
        return

    def reset_running_train_loss(self):
        self.running_train_loss = {}
        self.n_training_batches = 0
        self.sample_chosen_for_callback = False
        return

    def reset_running_val_loss(self):
        self.running_val_loss = {}
        self.n_validation_batches = 0
        self.sample_chosen_for_callback = False
        return

    def log_scalars(self, epoch=0, n_batches=0., model_label=None):
        train_loss = {key: item / self.n_training_batches for key, item in self.running_train_loss.items()}
        val_loss = {key: item / self.n_validation_batches for key, item in self.running_val_loss.items()}

        for key in train_loss:
            self.logger.add_scalar('Train/%s' % (key), train_loss[key], global_step=epoch)
        for key in val_loss:
            self.logger.add_scalar('Val/%s' % (key), val_loss[key], global_step=epoch)
        return

    def log_images(self, epoch=0):
        if epoch % self.logger_params['image_epoch_freq'] == 0:
            if 'prediction' in self.logger_params['image_callbacks']:
                self.draw_prediction_figure(
                    epoch,
                    self.first_validation_sample[self.input_type.split(',')[0]],
                    self.first_validation_sample['output'],
                    self.first_validation_sample['prediction'],
                    self.logger,
                )

            if 'vectorfield' in self.logger_params['image_callbacks']:
                self.draw_vectorfields_figure(
                    epoch,
                    self.first_validation_sample[self.input_type.split(',')[0]],
                    self.first_validation_sample['output'],
                    self.first_validation_sample['prediction'],
                    self.logger,
                )
            if 'hists' in self.logger_params['image_callbacks']:
                self.draw_force_hists_figure(
                    epoch,
                    self.first_validation_sample[self.input_type.split(',')[0]],
                    self.first_validation_sample['output'],
                    self.first_validation_sample['prediction'],
                    self.logger,
                )
        return

    def draw_vectorfields_figure(self, epoch, input, output, prediction, logger):
        colorscheme_dict = {
            'none': {
                'input': utils_plot.PositiveNorm(vmax=0.5, cmap='gray'),
                'output': utils_plot.SymmetricNorm(vmax=None),
                'prediction': utils_plot.SymmetricNorm(vmax=None),
            },
            'individually_normed': {
                'input': utils_plot.PositiveNorm(vmax=0.5, cmap='gray'),
                'output': utils_plot.SymmetricNorm(vmax='individual'),
                'prediction': utils_plot.SymmetricNorm(vmax='individual'),
            },
        }

        figscale = self.logger_params.get('figscale', 2)
        cscheme = colorscheme_dict[self.logger_params.get('predfig_cscheme', 'individually_normed')]

        nrows = input.shape[0]
        ncols = input.shape[1] + 2

        fig, ax = plt.subplots(nrows, ncols, figsize=(figscale * ncols, figscale * nrows), squeeze=False)

        with torch.no_grad():
            # Convert to CPU and NumPy once for the entire batch
            input_np = input.cpu().numpy() if torch.is_tensor(input) else input
            output_np = output.cpu().numpy() if torch.is_tensor(output) else output
            prediction_np = prediction.cpu().numpy() if torch.is_tensor(prediction) else prediction

            for b in range(nrows):
                mag_T = output_np[b][0] if self.angmag else np.linalg.norm(output_np[b], axis=0)
                mag_P = prediction_np[b][0] if self.angmag else np.linalg.norm(prediction_np[b], axis=0)

                ax[b][0].imshow(input_np[b][0] / input_np[b][0].max(), origin=\'lower\', **cscheme[\'input\'](input_np, b))

                ax[b][1].imshow(mag_T, origin=\'lower\', vmax=4, cmap=\'inferno\')
                ax[b][1].quiver(
                    *utils_plot.make_vector_field(*output_np[b], downsample=20, threshold=0.4, angmag=self.angmag),
                    color=\'w\',
                    width=0.003,
                    scale=20,
                )

                ax[b][2].imshow(mag_P, origin=\'lower\', vmax=4, cmap=\'inferno\')
                ax[b][2].quiver(
                    *utils_plot.make_vector_field(*prediction_np[b], downsample=20, threshold=0.4, angmag=self.angmag),
                    color=\'w\',
                    width=0.003,
                    scale=20,
                )

        for a in ax.flat: a.axis('off')

        ax[0][0].text(s='Input', **utils_plot.texttop, transform=ax[0][0].transAxes)
        ax[0][1].text(s='Target', **utils_plot.texttop, transform=ax[0][1].transAxes)
        ax[0][2].text(s='Prediction', **utils_plot.texttop, transform=ax[0][2].transAxes)

        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        logger.add_figure("vectorfields/model_%u" % self.index, fig, close=True, global_step=epoch)
        return

    def draw_prediction_figure(self, epoch, input, output, prediction, logger):
        colorscheme_dict = {
            'none': {
                'input': utils_plot.PositiveNorm(vmax=0.5, cmap='gray'),
                'output': utils_plot.SymmetricNorm(vmax=None),
                'prediction': utils_plot.SymmetricNorm(vmax=None),
            },
            'individually_normed': {
                'input': utils_plot.PositiveNorm(vmax=0.5, cmap='gray'),
                'output': utils_plot.SymmetricNorm(vmax='individual'),
                'prediction': utils_plot.SymmetricNorm(vmax='individual'),
            },
        }

        figscale = self.logger_params.get('figscale', 2)
        cscheme = colorscheme_dict[self.logger_params.get('predfig_cscheme', 'individually_norm        nrows = input.shape[0]
        ncols = 5 # Fixed to 5 columns as per plotting logic

        fig, ax = plt.subplots(nrows, ncols, figsize=(figscale * ncols, figscale * nrows), squeeze=False)

        with torch.no_grad():           # Convert to CPU and NumPy once for the entire batch
            input_np = input.cpu().numpy() if torch.is_tensor(input) else input
            output_np = output.cpu().numpy() if torch.is_tensor(output) else output
            prediction_np = prediction.cpu().numpy() if torch.is_tensor(prediction) else prediction

            for b in range(nrows):
                ax[b][0].imshow(input_np[b][0] / input_np[b][0].max(), origin=\'lower\', **cscheme[\'input\'](input_np, b))

                ax[b][1].imshow(output_np[b][0], origin=\'lower\', **cscheme[\'output\'](output_np, b, 0))
                ax[b][2].imshow(output_np[b][1], origin=\'lower\', **cscheme[\'output\'](output_np, b, 1))

                ax[b][3].imshow(prediction_np[b][0], origin=\'lower\', **cscheme[\'output\'](prediction_np, b, 0)) # Use prediction_np here
                ax[b][4].imshow(prediction_np[b][1], origin=\'lower\', **cscheme[\'output\'](prediction_np, b, 1)) # Use prediction_np hereor a in ax.flat: a.axis('off')

        ax[0][0].text(s='Input', **utils_plot.texttop, transform=ax[0][0].transAxes)
        ax[0][1].text(s='Target\n(Channel 0)', **utils_plot.texttop, transform=ax[0][1].transAxes)
        ax[0][2].text(s='Target\n(Channel 1)', **utils_plot.texttop, transform=ax[0][2].transAxes)
        ax[0][3].text(s='Prediction\n(Channel 0)', **utils_plot.texttop, transform=ax[0][3].transAxes)
        ax[0][4].text(s='Prediction\n(Channel 1)', **utils_plot.texttop, transform=ax[0][4].transAxes)

        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        logger.add_figure("predictions/model_%u" % self.index, fig, close=True, global_step=epoch)
        return

    def draw_force_hists_figure(self, epoch, input, output, prediction, logger):
        figscale = self.logger_params.get('figscale', 3)

        nrows = 1
        ncols = 3

        fig, ax = plt.subplots(1, 3, figsize=(figscale * ncols, figscale * nrows))

        Fmax = 6
        Fbins = np.linspace(0, Fmax, 101)
        cmap = 'GnBu'

        with torch.no_grad():
            hist_joint = np.zeros((len(Fbins) - 1, len(Fbins) - 1))

            # Convert to CPU and NumPy once for the entire batch
            input_np = input.cpu().numpy() if torch.is_tensor(input) else input
            output_np = output.cpu().numpy() if torch.is_tensor(output) else output
            prediction_np = prediction.cpu().numpy() if torch.is_tensor(prediction) else prediction

            for b in range(input_np.shape[0]): # Use input_np.shape[0]
                mag_T = output_np[b][0] if self.angmag else np.linalg.norm(output_np[b], axis=0)
                mag_P = prediction_np[b][0] if self.angmag else np.linalg.norm(prediction_np[b], axis=0)

                hist_joint += np.histogram2d(mag_T.ravel(), mag_P.ravel(), bins=(Fbins, Fbins))[0]

            hist_joint = hist_joint.T / np.sum(hist_joint)

            p_Fexp = np.sum(hist_joint, axis=1)
            # Handle division by zero for p_Fexp if it contains zeros
            hist_cond = np.zeros_like(hist_joint) # Initialize hist_cond with zeros
            non_zero_p_Fexp = p_Fexp != 0
            hist_cond[:, non_zero_p_Fexp] = hist_joint[:, non_zero_p_Fexp] / p_Fexp[non_zero_p_Fexp][None, :]

            extent = [Fbins.min(), Fbins.max(), Fbins.min(), Fbins.max()]

            cmap_joint = ax[0].imshow(
                hist_joint,
                origin='lower',
                extent=extent,
                cmap=cmap,
                norm=pltclr.SymLogNorm(linthresh=1.0, vmax=np.max(hist_joint) * 1e-3, vmin=0),
            )
            cmap_cond = ax[1].imshow(hist_cond, origin='lower', extent=extent, cmap=cmap, vmax=np.max(hist_cond) * 1e-2)

            for a in ax:
                xlim = a.get_xlim()
                a.plot(xlim, xlim, 'gray', ls=':')

            ax[2].semilogy(0.5 * (Fbins[1:] + Fbins[:-1]), np.sum(hist_joint, axis=0), label='$F_{exp}$',)
            ax[2].semilogy(0.5 * (Fbins[1:] + Fbins[:-1]), np.sum(hist_joint, axis=1), label='$F_{pred}$',)

        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        logger.add_figure("hists/model_%u" % self.index, fig, close=True, global_step=epoch)
        return
