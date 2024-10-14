
import torch 
import os
from utils import show_batch

def discriminator_step(discriminator, 
                       opt_d, 
                       generator, 
                       images, 
                       latent_dim, 
                       device):
  batch_size = images.shape[0]

  latent_vectors = torch.randn(size=(batch_size, latent_dim, 1, 1), device=device)
  fake_images = generator(latent_vectors)
  fake_predictions = discriminator(fake_images)
  fake_targets = torch.zeros(size= (batch_size, 1), device=device)
  fake_loss = torch.nn.functional.binary_cross_entropy(fake_predictions, fake_targets)

  real_predictions = discriminator(images)
  real_targets = torch.ones(size= (batch_size, 1), device=device)
  real_loss = torch.nn.functional.binary_cross_entropy(real_predictions, real_targets)

  loss = real_loss + fake_loss 
  opt_d.zero_grad()
  loss.backward()
  opt_d.step()

  return loss.detach().item()


def generator_step(generator, 
                   opt_g, 
                   discriminator, 
                   batch_size, 
                   latent_dim, 
                   device):

  latent_vectors = torch.randn(size=(batch_size, latent_dim, 1, 1), device=device)
  fake_images = generator(latent_vectors)
  fake_predictions = discriminator(fake_images)
  fake_targets = torch.ones(size= (batch_size, 1), device=device)
  loss = torch.nn.functional.binary_cross_entropy(fake_predictions, fake_targets)

  opt_g.zero_grad()
  loss.backward()
  opt_g.step()

  return loss.detach().item()


def train(discriminator, 
          generator, 
          train_dl, 
          batch_size, 
          latent_dim, 
          device, 
          num_epochs, 
          save_dir):
  os.makedirs(save_dir, exist_ok=True)
  fixed_latents = torch.randn(size=(batch_size, latent_dim, 1, 1), device=device)

  with torch.inference_mode():
    fixed_images = generator(fixed_latents)
    temp_name = f"generated_{0}"
    show_batch(fixed_images.cpu().detach(), nrow=8, save=True, save_name=save_dir+"/"+temp_name)

  opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
  opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

  for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1} of {num_epochs}")
    for xb, yb in train_dl:
      xb = xb.to(device)
      disc_loss = discriminator_step(discriminator, opt_d, generator, xb, latent_dim, device)
      gen_loss = generator_step(generator, opt_g, discriminator, batch_size, latent_dim, device)

    with torch.inference_mode():
      fixed_images = generator(fixed_latents)
      temp_name = f"generated_{epoch+1}"
      show_batch(fixed_images.cpu().detach(), nrow=8, save=True, save_name=save_dir+"/"+temp_name)

