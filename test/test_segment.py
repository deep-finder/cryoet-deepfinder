import os
import unittest
import shutil, tempfile
import numpy as np

from deepfinder.inference import Segment
from deepfinder.models import my_model


class TestSegment(unittest.TestCase):

    def setUp(self):
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create dummy model weights and save to temp dir:
        net = my_model(dim_in=88, Ncl=2)
        net.save(os.path.join(self.test_dir, 'net_weights_dummy.h5'))

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_segment(self):
        # Create dummy tomo:
        tomo = np.zeros((100,100,100), dtype=np.float16)

        # Initialize segmentation class:
        seg = Segment(
            Ncl=2,
            path_weights=os.path.join(self.test_dir, 'net_weights_dummy.h5'),
            patch_size=88,
        )

        # Launch segment:
        smap = seg.launch(tomo)


if __name__ == '__main__':
    unittest.main()
