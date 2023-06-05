"""
WIP. Latent Diffusion.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

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

        # TODO: look into stop gradient
        # commitment_loss = jnp.mean((jnp.stop_gradient(quantized) - x) ** 2)
        # codebook_loss = jnp.mean((quantized - jnp.stop_gradient(x)) ** 2)
        commitment_loss = jnp.mean((quantized - x) ** 2)
        codebook_loss = jnp.mean((quantized - x) ** 2)
        loss = self.beta * commitment_loss + codebook_loss

        # Straight-through estimator.
        # quantized = x + jnp.stop_gradient(quantized - x)
        return quantized, loss