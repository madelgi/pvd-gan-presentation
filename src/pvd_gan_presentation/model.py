"""Model implementation.
"""
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model


DEFAULT_GENERATOR = {
    # Architecture
    'input_shape': (2,),
    'h1_size': 16, 'h1_activation': LeakyReLU(alpha=0.2),
    'h2_size': 16, 'h2_activation': LeakyReLU(alpha=0.2),
    'output': 2, 'output_activation': 'linear',
}

DEFAULT_DISCRIMINATOR = {
    # Architecture
    'input_shape': (2,),
    'h1_size': 16, 'h1_activation': LeakyReLU(alpha=0.2),
    'h2_size': 16, 'h2_activation': LeakyReLU(alpha=0.2),
    'h3_size': 2, 'h3_activation': 'linear',
    'output': 1, 'output_activation': 'sigmoid',

    # Optimizer/loss
    'optimizer': 'rmsprop',
    'loss': 'binary_crossentropy'
}

class PVDGenerator:

    def __init__(self, model_params: dict = DEFAULT_GENERATOR):
        """Generator neural network.

        Arguments:
            model_params: A dictionary containing the model configuration. See the defaults above
                for format.
        """
        self.mp: dict = model_params
        self.model: Model = self.initialize_model()

    def initialize_model(self) -> Model:
        inputs = Input(shape=self.mp['input_shape'])
        hidden1 = Dense(self.mp['h1_size'], activation=self.mp['h1_activation'])(inputs)
        hidden2 = Dense(self.mp['h2_size'], activation=self.mp['h2_activation'])(hidden1)
        outputs = Dense(self.mp['output'], activation=self.mp['output_activation'])(hidden2)
        model = Model(inputs=inputs, outputs=outputs)
        return model


class PVDDiscriminator:

    def __init__(self, model_params: dict = DEFAULT_DISCRIMINATOR):
        """Discriminator neural network.

        Arguments:
            model_params: A dictionary containing the model configuration. See the defaults above
                for format.
        """
        self.mp: dict = model_params
        self.model: Model = self.initialize_model()

    def initialize_model(self) -> Model:
        inputs = Input(shape=self.mp['input_shape'])
        hidden1 = Dense(self.mp['h1_size'], activation=self.mp['h1_activation'])(inputs)
        hidden2 = Dense(self.mp['h2_size'], activation=self.mp['h2_activation'])(hidden1)
        hidden3 = Dense(self.mp['h3_size'], activation=self.mp['h3_activation'])(hidden2)
        outputs = Dense(self.mp['output'], activation=self.mp['output_activation'])(hidden3)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.mp['optimizer'], loss=self.mp['loss'], metrics=['accuracy'])
        return model


####################################################################################################
# MNIST (TODO)
####################################################################################################
class PVDGeneratorMnist:

    def __init__(self, model_params: dict = DEFAULT_GENERATOR):
        """Generator neural network.

        Arguments:
            model_params: A dictionary containing the model configuration. See the defaults above
                for format.
        """
        pass

    def initialize_model(self) -> Model:
        raise NotImplementedError("Not Implemented")


class PVDDiscriminatorMnist:

    def __init__(self, model_params: dict = DEFAULT_DISCRIMINATOR):
        """Discriminator neural network.

        Arguments:
            model_params: A dictionary containing the model configuration. See the defaults above
                for format.
        """
        pass

    def initialize_model(self) -> Model:
        raise NotImplementedError("Not Implemented")
