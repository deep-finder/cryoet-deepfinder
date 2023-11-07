import unittest
import numpy as np

from deepfinder.training import TargetBuilder
from deepfinder.inference import Cluster
from deepfinder.utils.eval import Evaluator

from test_utils import create_dummy_objl


class TestClustering(unittest.TestCase):

    def test_clustering(self):
        cradius = 3
        tomodim = (50, 100, 100)

        # Get dummy objl:
        #objl_true = create_dummy_objl(n_obj=20, n_obj_classes=2, tomodim=tomodim)

        objl_true = [
            {'tomo_idx': 0, 'label': 1, 'x': 10, 'y': 50, 'z': 25, 'phi': 0, 'psi': 0, 'the': 0,},
            {'tomo_idx': 0, 'label': 1, 'x': 20, 'y': 50, 'z': 25, 'phi': 0, 'psi': 0, 'the': 0,},
            {'tomo_idx': 0, 'label': 1, 'x': 30, 'y': 50, 'z': 25, 'phi': 0, 'psi': 0, 'the': 0,},
            {'tomo_idx': 0, 'label': 2, 'x': 40, 'y': 50, 'z': 25, 'phi': 0, 'psi': 0, 'the': 0,},
            {'tomo_idx': 0, 'label': 2, 'x': 50, 'y': 50, 'z': 25, 'phi': 0, 'psi': 0, 'the': 0,},
            {'tomo_idx': 0, 'label': 2, 'x': 60, 'y': 50, 'z': 25, 'phi': 0, 'psi': 0, 'the': 0,},
        ]

        # Create dummy label map:
        tbuild = TargetBuilder()
        lmap = tbuild.generate_with_spheres(
            objl=objl_true,
            target_array=np.zeros(tomodim),
            radius_list=[cradius, cradius]
        )

        # Initialize clustering task:
        clust = Cluster(clustRadius=cradius)

        # Launch clustering:
        objl_pred = clust.launch(lmap)

        # Eval:
        dset_true = {'tomo0': {'object_list': objl_true}}
        dset_pred = {'tomo0': {'object_list': objl_pred}}
        eval = Evaluator(dset_true, dset_pred, dist_thr=2).get_evaluation(score_thr=None)

        self.assertEqual(eval['global']['f1s']['1'], 1)  # f1 scores for all classes should be =1
        self.assertEqual(eval['global']['f1s']['2'], 1)


if __name__ == '__main__':
    unittest.main()