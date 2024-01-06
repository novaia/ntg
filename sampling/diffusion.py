from jax import lax
from jax import numpy as jnp

# TODO: figure out what this sampling method is called, it is implicit reverse diffusion
# since it skips steps but I'm pretty sure there is a more specific name.
def implicit(
    state, 
    num_images:int, 
    diffusion_steps:int, 
    image_width:int, 
    image_height:int, 
    channels:int,
    min_signal_rate:float,
    max_signal_rate:float,
    noise_clip:float,
    seed:int, 
):
    @jax.jit
    def inference_fn(state, noisy_images, diffusion_times):
        return lax.stop_gradient(
            state.apply_fn({'params': state.params}, noisy_images, diffusion_times)
        )
    
    initial_noise = jax.random.normal(
        jax.random.PRNGKey(seed), 
        shape=(num_images, image_height, image_width, channels)
    )
    initial_noise = jnp.clip(initial_noise, -noise_clip, noise_clip)
    step_size = 1.0 / diffusion_steps
    
    next_noisy_images = initial_noise
    for step in range(diffusion_steps):
        noisy_images = next_noisy_images
        
        diffusion_times = jnp.ones((num_images, 1, 1, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule(
            diffusion_times, min_signal_rate, max_signal_rate
        )
        pred_noises = inference_fn(state, noisy_images, noise_rates**2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule(
            next_diffusion_times, min_signal_rate, max_signal_rate
        )
        next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)
    return pred_images
