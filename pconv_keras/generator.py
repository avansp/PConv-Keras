from keras.preprocessing.image import ImageDataGenerator


class AugmentingDataGenerator(ImageDataGenerator):
    def flow_from_dataframe(self, csv_file, folder, *args, **kwargs):
        generator = super().flow_from_dataframe(
            csv_file,
            directory=folder,
            x_col='files',
            class_mode=None,
            *args, ** kwargs
        )
        while True:
            # get augmented image samples
            ori = next(generator)

            # yield
            yield ori

    def flow_from_directory(self, directory, mask_generator, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:
            # Get augmentend image samples
            ori = next(generator)

            # Get masks for each image sample
            mask = np.stack([
                mask_generator.sample(seed)
                for _ in range(ori.shape[0])], axis=0
            )

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori