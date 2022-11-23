import numpy as np
import deepfinder.utils.objl as ol


# Create dummy inputs:
def create_dummy_objl(n_obj=100, mono_class=True):
    objl = []
    for _ in range(n_obj):
        x = np.random.randint(0, 500)
        y = np.random.randint(0, 500)
        z = np.random.randint(0, 200)
        if mono_class:
            label = 1
        else:
            label = np.random.randint(1, 4)
        cluster_size = np.random.uniform(0, 1)
        objl = ol.add_obj(objl, label=label, coord=(z, y, x), cluster_size=cluster_size)
    return objl


def create_dummy_dset_for_evaluator(n_tomos=5, n_obj=100, mono_class=True):
    dset = {}
    for idx in range(n_tomos):
        key = 'tomo'+str(idx)
        dset[key] = {'object_list': create_dummy_objl(n_obj, mono_class)}
    return dset

