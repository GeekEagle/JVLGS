class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'SimGas':
            # return './datasets/kfold_SimGas/'
            return '/home/geek1234/datasets/kfold_marshall/'
        elif dataset == 'IGS-Few':
            # return './datasets/IGS-Few/'
            return '/home/geek1234/datasets/IGS-Few/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
