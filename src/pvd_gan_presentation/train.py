"""
Class controlling GAN training.
"""
import logging

from keras.losses import BinaryCrossentropy
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

from pvd_gan_presentation.model import PVDGenerator, PVDDiscriminator
from pvd_gan_presentation.data import sample_data


logger = logging.getLogger(__name__)


class PVDGAN:
    def __init__(self, generator: PVDGenerator = None, discriminator: PVDDiscriminator = None):
        if not (generator and discriminator):
            raise ValueError("Please pass a generator and discriminator to PVDTrainer.")

        self.g = generator
        self.d = discriminator
        self.gan = self.build_gan()

        # Plotting utilities
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot()

    def build_gan(self):
        z = Input(shape=self.g.mp['input_shape'])
        generated_out = self.g.model(z)
        self.d.model.trainable = False
        is_real = self.d.model(generated_out)

        combined = Model(z, is_real)
        combined.compile(loss=BinaryCrossentropy(from_logits=True), optimizer='rmsprop')
        return combined

    def train(self, num_iters: int = 20000, batch_size: int = 256, visualize: bool = False) -> None:
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        g_loss = None

        # Hold
        Z_batch = np.random.uniform(-1, 1, (batch_size, self.g.mp['input_shape'][0]))
        xdata, ydata = self.g.model.predict(Z_batch).T

        if visualize:
            initial_gen = self._ax.scatter(xdata, ydata)
            plt.ion()
            plt.show()

        for step in range(num_iters):
            # Generate noise and output for generator
            X_batch = sample_data(num_pts=batch_size)
            Z_batch = np.random.uniform(-1, 1, (batch_size, self.g.mp['input_shape'][0]))
            X_fake = self.g.model.predict(Z_batch)

            d_loss_real = self.d.model.train_on_batch(X_batch, valid)
            d_loss_fake = self.d.model.train_on_batch(X_fake, fake)

            # Discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # GAN Loss
            if step > 0 and step % 5 == 0:
                g_loss = self.gan.train_on_batch(Z_batch, valid)

            if step % 50 == 0 and visualize:
                xdata, ydata = X_fake.T
                self._ax.cla()
                self._ax.scatter(xdata, ydata)
                self._ax.set_ylim([0, 2500])
                self._ax.set_xlim([-50, 50])
                self._fig.canvas.draw()
                self._fig.canvas.flush_events()

            if step % 100 == 0:
                logger.error(f"Iteration {step} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")

    def test_generator(self, test_points: np.array) -> np.array:
        """Run the generator on the given input.
        """
        return self.g.model.predict_on_batch(test_points)



if __name__ == '__main__':
    gen = PVDGenerator()
    disc = PVDDiscriminator()
    trainer = PVDGAN(generator=gen, discriminator=disc)
    trainer.train(visualize=True)
