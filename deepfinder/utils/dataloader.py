import os
import warnings
import deepfinder.utils.objl as ol


class Dataloader:
    def __init__(self):
        self.path_data = []
        self.path_target = []
        self.objl_train = []
        self.objl_valid = []
        self.tomo_idx = 0

    def __call__(self, path_dset):
        path_train = os.path.join(path_dset, 'train')
        path_valid = os.path.join(path_dset, 'valid')

        if os.path.isdir(path_train):
            self.load_content(path_train)
        else:
            raise Exception('DeepFinder: train folder has not been found.')

        if os.path.isdir(path_valid):
            self.load_content(path_valid)
        else:
            #raise Warning('DeepFinder: valid folder has not been found.')
            print('DeepFinder: valid folder has not been found.')

        return self.path_data, self.path_target, self.objl_train, self.objl_valid

    def load_content(self, path):
        for fname in os.listdir(path):
            # if fname does not start with '.' (invisible temporary files) and end with '_objl.xml'
            if fname[0] is not '.' and fname.endswith('_objl.xml'):
                fprefix = fname[:-9]  # remove '_objl.xml'
                fname_tomo = fprefix + '.mrc'
                fname_target = fprefix + '_target.mrc'

                fname = os.path.join(path, fname)
                fname_tomo = os.path.join(path, fname_tomo)
                fname_target = os.path.join(path, fname_target)

                self.path_data.append(fname_tomo)
                self.path_target.append(fname_target)

                objl = ol.read(fname)
                for obj in objl:  # attribute tomo_idx to objl
                    obj['tomo_idx'] = self.tomo_idx

                if path.endswith('train'):
                    self.objl_train += objl
                elif path.endswith('valid'):
                    self.objl_valid += objl

                self.tomo_idx += 1


#path_dset = '/net/serpico-fs2/emoebel/cryo/shrec2021/localization/test_deepfinder/test_dataloader/data/'
#path_data, path_target, objl_train, objl_valid = Dataloader()(path_dset)