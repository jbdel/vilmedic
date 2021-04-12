import os


class InitBase(object):
    def __init__(self, opts):
        self.opts = opts
        self.ckpt_dir = os.path.join(self.opts.ckpt_dir, self.opts.name)


class Base(InitBase):
    def __init__(self, opts):
        super().__init__(opts)

    def start(self):
        raise NotImplementedError()
