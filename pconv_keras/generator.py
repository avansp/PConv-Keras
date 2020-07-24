from keras.preprocessing.image import ImageDataGenerator
import gc
import numpy as np
from copy import deepcopy


class AugmentingDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super(AugmentingDataGenerator, self).__init__(*args, **kwargs)
        self.generator = None
        self.mask_generator = None

    def generate(self, *args, **kwargs):
        assert self.generator is not None, "Please set the generator first."
        assert self.mask_generator is not None, "Please set the mask generator first."

        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:
            # Get augmentend image samples
            ori = next(self.generator)

            # Get masks for each image sample
            mask = np.stack([
                self.mask_generator.sample(seed)
                for _ in range(ori.shape[0])], axis=0
            )

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori

    def flow_from_dataframe(self, df, mask_generator, folder=None, *args, **kwargs):
        self.generator = super().flow_from_dataframe(df, directory=folder, class_mode=None, *args, **kwargs)
        self.mask_generator = mask_generator
        return self.generate(*args, **kwargs)

    def flow_from_directory(self, directory, mask_generator, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        self.mask_generator = mask_generator
        return self.generate(*args, **kwargs)