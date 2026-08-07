import yaml
import random
import argparse
import os
import torch
import librosa
from tqdm import tqdm
from diffusers import DDIMScheduler
from model.solospeech.conditioners import SoloSpeech_TSE
from model.solospeech.conditioners import SoloSpeech_TSR
from utils import save_audio
import shutil
from vae_modules.autoencoder_wrapper import Autoencoder
import pandas as pd
from speechbrain.pretrained.interfaces import Pretrained
from corrector.fastgeco.model import ScoreModel
from corrector.geco.util.other import pad_spec

class Encoder(Pretrained):

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings,
                torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--output-dir', type=str, default='')
parser.add_argument('--test_dir', type=str, default='')
# pre-trained model path
parser.add_argument('--autoencoder-path', type=str, default='./pretrained_models/vae.ckpt')
parser.add_argument('--vae-rate', type=int, default=50)
parser.add_argument('--eta', type=int, default=0)

parser.add_argument("--num_infer_steps", type=int, default=200)
# model configs
parser.add_argument('--vae-config', type=str, default='./pretrained_models/config.json')
parser.add_argument('--tse-config', type=str, default='./config/SoloSpeech-tse-base.yaml')
parser.add_argument('--tse-ckpt', type=str, default='')
parser.add_argument('--tsr-config', type=str, default='./config/SoloSpeech-tsr-base.yaml')
parser.add_argument('--tsr-ckpt', type=str, default='')
parser.add_argument("--geco-ckpt", type=str, default='')
parser.add_argument("--reverse_starting_point", type=float, default=0.5, help="Starting point for the geco reverse SDE.")
parser.add_argument("--N", type=int, default=1, help="Number of geco reverse steps")
parser.add_argument('--sample-rate', type=int, default=16000)
parser.add_argument('--debug', type=bool, default=False)
# log and random seed
parser.add_argument('--random-seed', type=int, default=2024)
args = parser.parse_args()

with open(args.tse_config, 'r') as fp:
    args.tse_config = yaml.safe_load(fp)

with open(args.tsr_config, 'r') as fp:
    args.tsr_config = yaml.safe_load(fp)

args.v_prediction = args.tse_config["ddim"]["v_prediction"]


@torch.no_grad()
def sample_diffusion(tse_model, tsr_model, autoencoder, std, scheduler, device,
                     mixture=None, reference=None, lengths=None, reference_lengths=None, 
                     ddim_steps=50, eta=0, seed=2023
                     ):

    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(ddim_steps)
    tse_pred = torch.randn(mixture.shape, generator=generator, device=device)
    tsr_pred = torch.randn(mixture.shape, generator=generator, device=device)

    for t in scheduler.timesteps:
        tse_pred = scheduler.scale_model_input(tse_pred, t)
        model_output, _ = tse_model(
            x=tse_pred, 
            timesteps=t, 
            mixture=mixture, 
            reference=reference, 
            x_len=lengths, 
            ref_len=reference_lengths
            )
        tse_pred = scheduler.step(model_output=model_output, timestep=t, sample=tse_pred,
                                eta=eta, generator=generator).prev_sample
    
    for t in scheduler.timesteps:
        tsr_pred = scheduler.scale_model_input(tsr_pred, t)
        model_output, _ = tsr_model(
            x=tsr_pred, 
            timesteps=t, 
            mixture=mixture, 
            reference=tse_pred, 
            x_len=lengths, 
            )
        tsr_pred = scheduler.step(model_output=model_output, timestep=t, sample=tsr_pred,
                                eta=eta, generator=generator).prev_sample

    tse_pred = autoencoder(embedding=tse_pred.transpose(2,1), std=std).squeeze(1)
    tsr_pred = autoencoder(embedding=tsr_pred.transpose(2,1), std=std).squeeze(1)

    return tse_pred, tsr_pred



if __name__ == '__main__':

    os.makedirs(args.output_dir, exist_ok=True)
    autoencoder = Autoencoder(args.autoencoder_path, args.vae_config, 'stft_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.to(args.device)

    tse_model = SoloSpeech_TSE(
        args.tse_config['diffwrap']['UDiT'],
        args.tse_config['diffwrap']['ViT'],
    ).to(args.device)
    tse_model.load_state_dict(torch.load(args.tse_ckpt)['model'])
    tse_model.eval()

    total = sum([param.nelement() for param in tse_model.parameters()])
    print("TSE Number of parameter: %.2fM" % (total / 1e6))

    tsr_model = SoloSpeech_TSR(
        args.tsr_config['diffwrap']['UDiT']
    ).to(args.device)
    tsr_model.load_state_dict(torch.load(args.tsr_ckpt)['model'])
    tsr_model.eval()

    total = sum([param.nelement() for param in tsr_model.parameters()])
    print("TSR Number of parameter: %.2fM" % (total / 1e6))
    
    geco_model = ScoreModel.load_from_checkpoint(
        args.geco_ckpt,
        batch_size=1, num_workers=0, kwargs=dict(gpu=False)
    )
    geco_model.eval(no_ema=False)
    geco_model.cuda()

    ecapatdnn_model = Encoder.from_hparams(source="yangwang825/ecapa-tdnn-vox2")
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    
    noise_scheduler = DDIMScheduler(**args.tse_config["ddim"]['diffusers'])
    
    # these steps reset dtype of noise_scheduler params
    latents = torch.randn((1, 128, 128),
                          device=args.device)
    noise = torch.randn(latents.shape).to(latents.device)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (noise.shape[0],),
                              device=latents.device).long()
    _ = noise_scheduler.add_noise(latents, noise, timesteps)
    
    mix_paths = []
    for root, dirs, files in os.walk(args.test_dir):
        for file in files:
            if file.endswith('_mix.wav'):
                mix_paths.append(os.path.join(root, file))
    
    if args.debug:
        mix_paths = mix_paths[:10]
    
    meta_data = []
    for mix_path in mix_paths:
        mixture_path = mix_path
        source_path = mix_path.replace("_mix.wav","_ref.wav")
        enroll_path = mix_path.replace("_mix.wav","_enrollment.wav")
        meta_data.append([mixture_path, source_path, enroll_path]) 
    
    SEGMENT_LEN = 16000 * 20
    for i in tqdm(range(len(meta_data))):
        savename = meta_data[i][0].split('/')[-1].replace("_mix.wav","")
        mixture, _ = librosa.load(meta_data[i][0], sr=16000)
        mixture = mixture[:320000]
        reference, _ = librosa.load(meta_data[i][2], sr=16000)
        reference_wav = reference
        reference = torch.tensor(reference).unsqueeze(0).to(args.device)
        with torch.no_grad():
            reference, _ = autoencoder(audio=reference.unsqueeze(1))
            reference_lengths = torch.LongTensor([reference.shape[-1]]).to(args.device)
        
        preds = []
        x_hats = []
        for start in range(0, len(mixture), SEGMENT_LEN):
            end = min(start + SEGMENT_LEN, len(mixture))
            with torch.no_grad():
                mixture_input = torch.tensor(mixture[start:end]).unsqueeze(0).to(args.device)
                mixture_wav = mixture_input
                mixture_input, std = autoencoder(audio=mixture_input.unsqueeze(1))
                lengths = torch.LongTensor([mixture_input.shape[-1]]).to(args.device)
                
            tse_pred, tsr_pred = sample_diffusion(tse_model, tsr_model, autoencoder, std, noise_scheduler, args.device, mixture_input.transpose(2,1), reference.transpose(2,1), lengths, reference_lengths, ddim_steps=args.num_infer_steps, eta=args.eta, seed=args.random_seed)
            
            ecapatdnn_embedding1 = ecapatdnn_model.encode_batch(tse_pred.squeeze()).squeeze()
            ecapatdnn_embedding2 = ecapatdnn_model.encode_batch(tsr_pred.squeeze()).squeeze()
            ecapatdnn_embedding3 = ecapatdnn_model.encode_batch(torch.tensor(reference_wav)).squeeze()
            sim1 = cosine_sim(ecapatdnn_embedding1, ecapatdnn_embedding3).item()
            sim2 = cosine_sim(ecapatdnn_embedding2, ecapatdnn_embedding3).item()
            pred = tse_pred if sim1 > sim2 else tsr_pred
            preds.append(pred)
            
            # corrector
            min_leng = min(pred.shape[-1], mixture_wav.shape[-1])
            x = pred[...,:min_leng]
            m = mixture_wav[...,:min_leng]
            norm_factor = m.abs().max()
            x = x / norm_factor
            m = m / norm_factor 
            X = torch.unsqueeze(geco_model._forward_transform(geco_model._stft(x.cuda())), 0)
            X = pad_spec(X)
            M = torch.unsqueeze(geco_model._forward_transform(geco_model._stft(m.cuda())), 0)
            M = pad_spec(M)
            timesteps = torch.linspace(args.reverse_starting_point, 0.03, args.N, device=M.device)
            std = geco_model.sde._std(args.reverse_starting_point*torch.ones((M.shape[0],), device=M.device))
            z = torch.randn_like(M)
            X_t = M + z * std[:, None, None, None]
            
            #reverse steps by Euler Maruyama
            for idx in range(len(timesteps)):
                t = timesteps[idx]
                if idx != len(timesteps) - 1:
                    dt = t - timesteps[idx+1]
                else:
                    dt = timesteps[-1]
                with torch.no_grad():
                    #take Euler step here
                    f, g = geco_model.sde.sde(X_t, t, M)
                    vec_t = torch.ones(M.shape[0], device=M.device) * t 
                    mean_x_tm1 = X_t - (f - g**2*geco_model.forward(X_t, vec_t, M, X, vec_t[:,None,None,None]))*dt #mean of x t minus 1 = mu(x_{t-1})
                    if idx == len(timesteps) - 1: #output
                        X_t = mean_x_tm1 
                        break
                    z = torch.randn_like(X) 
                    #Euler Maruyama
                    X_t = mean_x_tm1 + z*g*torch.sqrt(dt)

            sample = X_t
            sample = sample.squeeze()
            x_hat = geco_model.to_audio(sample.squeeze(), min_leng)
            x_hat = x_hat * norm_factor / x_hat.abs().max()
            x_hat = x_hat.detach().cpu()
            x_hats.append(x_hat)
    
        pred = torch.cat(preds, dim=-1)
        x_hat = torch.cat(x_hats, dim=-1)
        

        savename = meta_data[i][0].split('/')[-1].split('.wav')[0]
        shutil.copyfile(meta_data[i][0], f'{args.output_dir}/{savename}_mix.wav')
        # shutil.copyfile(meta_data[i][1], f'{args.output_dir}/{savename}_ref.wav')
        shutil.copyfile(meta_data[i][2], f'{args.output_dir}/{savename}_enrollment.wav')
        save_audio(f'{args.output_dir}/{savename}_pred.wav', 16000, pred)
        save_audio(f'{args.output_dir}/{savename}_final.wav', 16000, x_hat)
