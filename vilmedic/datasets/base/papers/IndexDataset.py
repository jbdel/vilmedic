from ... import *


class ActualIndexDataset:
    """
    Dataset wrapper that returns the index of the samples
    """
    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {
                **super(ActualIndexDataset, self).get_collate_fn()(batch),
                "index": [s['index'] for s in batch]
            }
            return collated

        return collate_fn

    def __getitem__(self, index):
        return {**super(ActualIndexDataset, self).__getitem__(index), 'index': index}

    def __repr__(self):
        return "IndexDataset with original dataset being: \n" + \
               super(ActualIndexDataset, self).__repr__()


def IndexDataset(dataset, **kwargs):
    return type('ActualIndexDataset', (ActualIndexDataset, eval(dataset), object), {})(**kwargs)
