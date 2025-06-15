class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'SimGas':
            return './datasets/kfold_SimGas/'
        elif dataset == 'IGS-Few':
            return './datasets/IGS-Few/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
