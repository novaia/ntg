import jax
from jax import lax
import jax.numpy as jnp

#@jax.jit
def reverse_diffusion(
    apply_fn, 
    params,
    batch_stats,
    num_images, 
    diffusion_steps, 
    image_width, 
    image_height, 
    channels, 
    diffusion_schedule_fn,
    seed, 
    initial_noise = None,
):
    if initial_noise == None:
        initial_noise = jax.random.normal(
            jax.random.PRNGKey(seed), 
            shape=(num_images, image_height, image_width, channels)
        )
    step_size = 1.0 / diffusion_steps
    
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images
        
        diffusion_times = jnp.ones((num_images, 1, 1, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule_fn(diffusion_times)
        #pred_noises = apply_fn(params, [noisy_images, noise_rates**2], mutable=False)
        pred_noises = lax.stop_gradient(
            apply_fn(
                {'params': params, 'batch_stats': batch_stats}, 
                [noisy_images, noise_rates**2],
                train=False,    
            )
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule_fn(next_diffusion_times)
        next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
        
    return pred_images