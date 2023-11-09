import os
import numpy as np
import deepfinder.utils.objl as ol
import deepfinder.utils.common as cm


# Create dummy inputs:
def create_dummy_objl(n_obj=100, n_obj_classes=1, tomodim=(200, 500, 500)):
    objl = []
    for _ in range(n_obj):
        x = np.random.randint(0, tomodim[2])
        y = np.random.randint(0, tomodim[1])
        z = np.random.randint(0, tomodim[0])
        label = np.random.randint(low=1, high=n_obj_classes+1)
        cluster_size = np.random.randint(low=1, high=100)
        objl = ol.add_obj(objl, label=label, coord=(z, y, x), cluster_size=cluster_size)
    return objl


def create_dummy_dset_for_evaluator(n_tomos=5, n_obj=100, n_obj_classes=1):
    dset = {}
    for idx in range(n_tomos):
        key = 'tomo'+str(idx)
        dset[key] = {'object_list': create_dummy_objl(n_obj, n_obj_classes)}
    return dset


def create_dummy_dset_for_train(path, n_tomos_train=2, n_tomos_valid=1, tomodim=(50, 100, 100), n_obj_classes=2):
    path_train = os.path.join(path, 'train')
    path_valid = os.path.join(path, 'valid')

    if not os.path.isdir(path_train): os.mkdir(path_train)
    if not os.path.isdir(path_valid): os.mkdir(path_valid)

    for idx in range(n_tomos_train):
        create_tomo_target_objl(path=path_train, prefix='tomo'+str(idx), tomodim=tomodim, n_obj_classes=n_obj_classes)

    for idx in range(n_tomos_valid):
        create_tomo_target_objl(path=path_valid, prefix='tomo'+str(idx))


def create_tomo_target_objl(path, prefix, tomodim=(50, 100, 100), n_obj_classes=2):
    tomo = np.random.rand(tomodim[0], tomodim[1], tomodim[2]).astype(np.float16)
    target = np.random.randint(low=0, high=n_obj_classes+1, size=tomodim, dtype=np.int8)
    objl = create_dummy_objl(n_obj=100, n_obj_classes=n_obj_classes)

    fname_tomo = os.path.join(path, prefix+'.mrc')
    fname_target = os.path.join(path, prefix+'_target.mrc')
    fname_objl = os.path.join(path, prefix+'_objl.xml')

    cm.write_array(tomo, filename=fname_tomo)
    cm.write_array(target, filename=fname_target)
    ol.write(objl, filename=fname_objl)