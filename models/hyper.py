"""
WIP. Latent Diffusion.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax import lax

# Translated from Keras docs: https://keras.io/examples/generative/vq_vae/
class VectorQuantize(nn.Module):
    embedding_dim: int
    num_embeddings = int
    beta: float

    @nn.compact
    def __call__(self, x):
        embeddings = self.param(
            'embeddings', 
            nn.initializers.uniform(), 
            (self.embedding_dim, self.num_embeddings)
        )

        input_shape = x.input_shape
        flattened = jnp.reshape(x, [-1, self.embedding_dim])

        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = jnp.matmul(flattened, embeddings)
        distances = (
            jnp.sum(flattened ** 2, axis=1, keepdims=True)
            + jnp.sum(embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = jnp.argmin(distances, axis=1)

        encodings = jnp.eye(self.num_embeddings)[encoding_indices]
        quantized = jnp.matmul(encodings, embeddings.T)
        quantized = jnp.reshape(quantized, input_shape)

        commitment_loss = jnp.mean((lax.stop_gradient(quantized) - x) ** 2)
        codebook_loss = jnp.mean((quantized - lax.stop_gradient(x)) ** 2)
        loss = self.beta * commitment_loss + codebook_loss

        # Straight-through estimator.
        quantized = x + lax.stop_gradient(quantized - x)
        return quantized, loss

# Reference: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
class Encoder(nn.Module):
    channels: int
    z_channels: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        # conv in
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)

        # downsampling
        

        # conv out
        x = nn.Conv(self.z_channels, kernel_size=(3, 3))(x)

        x = nn.Conv(self.z_channels, self.embed_dim, 1)(x)
    
class Decoder(nn.Module):
    z_channels: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        x, quant_loss = VectorQuantize(self.embed_dim, self.z_channels, beta=0.25)(x)
        x = nn.Conv(self.embed_dim, self.z_channels, 1)(x)
        return x, quant_loss

class VQGAN(nn.Module):
    z_channels: int
    embed_dim: int

    def setup(self):    
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv(self.z_channels, self.embed_dim, 1)
        self.quantize = VectorQuantize(self.embed_dim, self.z_channels, beta=0.25)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quantized, quantized_loss = self.quantize(h)
        return quantized, quantized_loss
    
    def encode_to_prequnt(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h
    
    def decode():
