import os
import unittest
import shutil, tempfile
import numpy as np

from deepfinder.training import TargetBuilder

from test_utils import create_dummy_objl


class TestTargetBuilder(unittest.TestCase):

    def test_target_builder(self):
        # Set initial volume: can be used to add segmented structures to target (e.g. membranes).
        # If not, simply initialize with empty volume (zero values)
        tomodim = (50, 100, 100)
        initial_vol = np.zeros(tomodim)

        # Get dummy objl:
        objl = create_dummy_objl(n_obj=20, n_obj_classes=2, tomodim=tomodim)

        radius_list = [3, 6]

        tbuild = TargetBuilder()
        target = tbuild.generate_with_spheres(objl, initial_vol, radius_list)

        self.assertEqual(list(np.unique(target)), [0, 1, 2])  # basic test: unique values of target should 0,1,2


if __name__ == '__main__':
    unittest.main()
