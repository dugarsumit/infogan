import numpy as np


class Dataset(object):
    def __init__(self, images, labels = None):
        self._images = images
        self._labels = labels
        if images is not None:
            self._num_samples = images.shape[0]
            self._num_examples = images.shape[0]
            self._index_in_epoch = self._num_examples
        self._epochs_completed = -1
        # shuffle on first run

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_samples(self):
        return self._num_samples

    """
    def next_random_batch(self, batch_size, replace = True):
        mask = np.random.choice(range(self.num_samples), batch_size, replace = replace)
        if self.labels is None:
            return self.images[mask]
        else:
            return self.images[mask], self.labels[mask]
    """

    def next_random_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            return self._images[start:end], self._labels[start:end]


    def next_supervised_random_batch(self, batch_size):
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        num_of_diff_labels = 10
        for cat in range(num_of_diff_labels):
            ids = np.where(self._labels == cat)[0]
            np.random.shuffle(ids)
            num_samples_per_cat = int(batch_size/num_of_diff_labels)
            sup_images.extend(self._images[ids[:num_samples_per_cat]])
            sup_labels.extend(self._labels[ids[:num_samples_per_cat]])
        np.random.set_state(rnd_state)
        return sup_images, sup_labels