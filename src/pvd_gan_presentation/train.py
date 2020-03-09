"""
Class controlling GAN training.
"""
import logging

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

from pvd_gan_presentation.model import PVDGenerator, PVDDiscriminator
from pvd_gan_presentation.data import sample_data


logger = logging.getLogger(__name__)


class PVDTrainer:

    def __init__(self, generator: PVDGenerator = None, discriminator: PVDDiscriminator = None):
        if not (generator and discriminator):
            raise ValueError("Please pass a generator and discriminator to PVDTrainer.")

        self.g = generator
        self.d = discriminator
        self.gan = self.build_gan()

    def build_gan(self):
        z = Input(shape=self.g.mp['input_shape'])
        out = self.g.model(z)
        self.d.model.trainable = False
        is_real = self.d.model(out)

        combined = Model(z, is_real)
        combined.compile(loss='binary_crossentropy', optimizer='rmsprop')
        return combined

    def train(self, num_iters: int = 10000, batch_size: int = 128) -> None:
        X_train = sample_data()
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for i, epoch in enumerate(range(num_iters)):
            # Generate noise and output for generator
            noise = np.random.normal(0, 1, (batch_size, self.g.mp['input_shape'][0]))
            X_fake = self.g.model.predict(noise)

            # Generate real pts for discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            X_batch = X_train[idx]

            # Discriminator loss
            d_loss_real = self.d.model.train_on_batch(X_batch, valid)
            d_loss_fake = self.d.model.train_on_batch(X_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Generator loss
            noise = np.random.normal(0, 1, (batch_size, self.g.mp['input_shape'][0]))
            X_fake = self.g.model.predict(noise)
            g_loss = self.gan.train_on_batch(X_fake, valid)
            if i % 100 == 0:
                logger.error(f"Iteration {epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")


if __name__ == '__main__':
    gen = PVDGenerator()
    disc = PVDDiscriminator()

    trainer = PVDTrainer(generator=gen, discriminator=disc)
    trainer.train()
