from models.vae3d import IV_VAE
from decord import VideoReader, cpu
import torch
import os
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
from fire import Fire

@torch.no_grad()
def main(video_path='./video/ori.mp4', save_path='video/gen.mp4', height=720, width=1280, z_dim=16, dim=64):
    # {ivvae_z4_dim64, ivvae_z8_dim64, ivvae_z16_dim64, ivvae_z16_dim96} 
    vae3d = IV_VAE(z_dim, dim).to(torch.bfloat16)
    vae3d.requires_grad_(False)

    transform = transforms.Compose([
        transforms.Resize(size=(height,width))
    ])

    vae3d = vae3d.cuda()
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    video_reader = VideoReader(video_path, ctx=cpu(0))

    fps = video_reader.get_avg_fps() 
    video = video_reader.get_batch(list(range(len(video_reader)))).asnumpy() 
    video = rearrange(torch.tensor(video),'t h w c -> t c h w') 
    video = transform(video) 
    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0).to(torch.bfloat16) 
    
    frame_end = 121 #1 + (len(video_reader) -1) // 4 * 4  
    video = video / 127.5 - 1.0 
    video= video[:,:,:frame_end,:,:]  
    video = video.cuda() 
    print(f'Shape of input video: {video.shape}')
    
    latent = vae3d.encode(video) 
    print(f'Shape of video latent: {latent.shape}') 
    
    results = vae3d.decode(latent)

    results = rearrange(results.squeeze(0), 'c t h w -> t h w c') 
    results = (torch.clamp(results,-1.0,1.0) + 1.0) * 127.5
    results = results.to('cpu', dtype=torch.uint8)

    write_video(save_path, results,fps=fps,options={'crf': '10'})

if __name__ == '__main__':
    main()