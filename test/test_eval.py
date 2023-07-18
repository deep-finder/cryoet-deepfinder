#import sys
#sys.path.append('../../deepfinder/')

import numpy as np
import deepfinder.utils.objl as ol
import deepfinder.utils.eval as ev

from test_utils import create_dummy_objl, create_dummy_dset_for_evaluator

import unittest
import copy


# # Create dummy inputs:
# def create_dummy_objl(n_obj=100, mono_class=True):
#     objl = []
#     for _ in range(n_obj):
#         x = np.random.randint(0, 500)
#         y = np.random.randint(0, 500)
#         z = np.random.randint(0, 200)
#         if mono_class:
#             label = 1
#         else:
#             label = np.random.randint(1, 4)
#         cluster_size = np.random.uniform(0, 1)
#         objl = ol.add_obj(objl, label=label, coord=(z, y, x), cluster_size=cluster_size)
#     return objl
#
#
# def create_dummy_data_set(n_tomos=5, n_obj=100, mono_class=True):
#     dset = {}
#     for idx in range(n_tomos):
#         key = 'tomo'+str(idx)
#         dset[key] = {'object_list': create_dummy_objl(n_obj, mono_class)}
#     return dset


class TestEvaluator(unittest.TestCase):

    def test_identity(self):  # here we test dset_true to itself. Should give perfect scores
        dset_true = create_dummy_dset_for_evaluator(n_tomos=5, n_obj=100, n_obj_classes=3)
        dset_pred = dset_true

        detect_eval = ev.Evaluator(dset_true, dset_pred, dist_thr=0).get_evaluation(score_thr=None)

        self.assertEqual(detect_eval['global']['f1s'][1], 1)  # f1 scores for all classes should be =1
        self.assertEqual(detect_eval['global']['f1s'][2], 1)
        self.assertEqual(detect_eval['global']['f1s'][3], 1)

        self.assertEqual(
            len(detect_eval['tomo0']['objl_tp']),  # n_tp should be = n_true
            len(detect_eval['tomo0']['objl_true']),
        )
        self.assertEqual(len(detect_eval['tomo0']['objl_fp']), 0)  # n_fp should be 0
        self.assertEqual(len(detect_eval['tomo0']['objl_fn']), 0)  # n_fn should be 0

    def test_recall(self):
        dset_true = create_dummy_dset_for_evaluator(n_tomos=1, n_obj=100, n_obj_classes=1)
        dset_pred = copy.deepcopy(dset_true)

        # Delete 10 elements from pred:
        n_delete = 10
        for _ in range(n_delete):
            del dset_pred['tomo0']['object_list'][0]

        detect_eval = ev.Evaluator(dset_true, dset_pred, dist_thr=0).get_evaluation(score_thr=None)

        # Hence recall should be 0.9 and precision should be 1
        self.assertEqual(detect_eval['tomo0']['rec'][1], 0.9)
        self.assertEqual(detect_eval['tomo0']['pre'][1], 1.0)

        # Also, following should be true: n_tp=90, n_fp=0, n_fn=10:
        self.assertEqual(len(detect_eval['tomo0']['objl_tp']), 90)
        self.assertEqual(len(detect_eval['tomo0']['objl_fp']), 0)
        self.assertEqual(len(detect_eval['tomo0']['objl_fn']), 10)

        # Global scores should also be as follow:
        self.assertEqual(detect_eval['global']['rec'][1], 0.9)
        self.assertEqual(detect_eval['global']['pre'][1], 1.0)

    def test_precision(self):
        dset_true = create_dummy_dset_for_evaluator(n_tomos=1, n_obj=100, n_obj_classes=1)
        dset_pred = copy.deepcopy(dset_true)

        # Delete 10 elements from true:
        n_delete = 10
        for _ in range(n_delete):
            del dset_true['tomo0']['object_list'][0]

        detect_eval = ev.Evaluator(dset_true, dset_pred, dist_thr=0).get_evaluation(score_thr=None)

        # Hence recall should be 1 and precision should be 0.9
        self.assertEqual(detect_eval['tomo0']['rec'][1], 1.0)
        self.assertEqual(detect_eval['tomo0']['pre'][1], 0.9)

        # Also, following should be true: n_tp=90, n_fp=10, n_fn=0:
        self.assertEqual(len(detect_eval['tomo0']['objl_tp']), 90)
        self.assertEqual(len(detect_eval['tomo0']['objl_fp']), 10)
        self.assertEqual(len(detect_eval['tomo0']['objl_fn']), 0)

        # Global scores should also be as follow:
        self.assertEqual(detect_eval['global']['rec'][1], 1.0)
        self.assertEqual(detect_eval['global']['pre'][1], 0.9)

if __name__ == '__main__':
    unittest.main()
