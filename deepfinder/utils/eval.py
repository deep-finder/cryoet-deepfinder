import numpy as np
from sklearn.metrics import pairwise_distances
from pycm import ConfusionMatrix
from . import objl as ol
import copy

class Evaluator:

    def __init__(self, dset_true, dset_pred, dist_thr):
        if dset_true.keys() != dset_pred.keys():
            raise Exception('Data sets do not have the same keys')
        self.dset_true = dset_true
        self.dset_pred = dset_pred
        self.dist_thr = dist_thr

        self.tomoid_keys = dset_true.keys()

        # detect_eval is a dict who stores, per tomo, all info related to the evaluation of the detections
        self.detect_eval = {}

        # Initialize detect_eval:
        for tomoid in dset_true.keys():
            self.detect_eval[tomoid] = {
                'objl_true': None,
                'objl_pred': None,
                'dmat': None,
                'cmat': None,
                'objl_tp': None,
                'objl_fp': None,
                'objl_fn': None,
                'pre': None,
                'rec': None,
                'f1s': None,
            }
        self.detect_eval['global'] = {
            'cmat': None,
            'pre': None,
            'rec': None,
            'f1s': None,
            'n_tp': None,
            'n_fp': None,
            'n_fn': None,
        }

        # detect_eval_list stores all the detect_eval w.r.t score_thr_list
        self.detect_eval_list = []

    def get_evaluation_wrt_detection_score(self, score_thr_list):
        for score_thr in score_thr_list:
            self.get_evaluation(score_thr)
            self.detect_eval_list.append(self.detect_eval)

        return self.detect_eval_list

    def get_evaluation(self, score_thr=None):
        for tomoid in self.tomoid_keys:
            objl_true = self.dset_true[tomoid]['object_list']
            objl_pred = self.dset_pred[tomoid]['object_list']
            if score_thr is not None:
                objl_pred = ol.above_thr(objl_pred, score_thr)
            self.detect_eval[tomoid]['objl_true'] = objl_true
            self.detect_eval[tomoid]['objl_pred'] = objl_pred

            self.get_distance_matrix(tomoid)

        self.get_confusion_matrix()
        self.get_metrics()

        return self.detect_eval

    def get_distance_matrix(self, tomoid):
        # Prepare data points for pairwise_distances:
        objl_true = self.detect_eval[tomoid]['objl_true']
        objl_pred = self.detect_eval[tomoid]['objl_pred']

        coords_true = np.zeros((len(objl_true), 3))
        coords_pred = np.zeros((len(objl_pred), 3))
        for idx, obj in enumerate(objl_true):
            coords_true[idx, 0] = obj['y']
            coords_true[idx, 1] = obj['x']
            coords_true[idx, 2] = obj['z']
        for idx, obj in enumerate(objl_pred):
            coords_pred[idx, 0] = obj['y']
            coords_pred[idx, 1] = obj['x']
            coords_pred[idx, 2] = obj['z']

        # Compute pairwise distances:
        if len(objl_pred) is not 0:  # only compute dmat if something has been detected
            dmat = pairwise_distances(coords_true, coords_pred, metric='euclidean')
        else:  # if no detections then dmat is not defined (and later on all scores are 0)
            dmat = None

        self.detect_eval[tomoid]['dmat'] = dmat

    def get_confusion_matrix(self):
        y_true_global = []
        y_pred_global = []
        for tomoid in self.tomoid_keys:
            dmat = self.detect_eval[tomoid]['dmat']
            objl_true = self.detect_eval[tomoid]['objl_true'].copy()
            objl_pred = self.detect_eval[tomoid]['objl_pred'].copy()
            if dmat is not None:  # i.e., if something has been detected
                # The aim of this part is to construct objl_pred_corresp and objl_true_corresp
                # from objl_pred and objl_true, such that objl_pred_corresp[i] matches objl_true_corresp[i]
                # (matches = same position with a tolerated position error of self.dist_thr)

                # Initialize objl_pred_corresp
                objl_pred_corresp = [None for _ in range(len(objl_true))]
                objl_true_corresp = objl_true.copy()

                # Correspondence matrix where '1' means that an entry from pred is situated at a distance <=dist_thr
                # to an entry of true:
                corresp_mat = dmat <= self.dist_thr

                pred_match_idx_list = []  # this list stores all the entries in objl_pred that have a match in objl_true
                for idx, obj in enumerate(objl_true_corresp):
                    indices = np.nonzero(corresp_mat[idx, :])[0]
                    if len(indices) == 1:  # tp only if 1 corresponence (necessary in case 1 true matches several pred)
                        objl_pred_corresp[idx] = objl_pred[indices[0]]
                        pred_match_idx_list.append(indices[0])
                    elif len(indices) < 1 or len(indices) > 1:  # if no correspondence or multiple correspondence, then fp
                        obj_fp = copy.deepcopy(obj)  # necessary else the labels in objl_true get modified
                        obj_fp['label'] = 0  # fp, therefore label is 0
                        objl_pred_corresp[idx] = obj_fp
                    # elif len(indices) > 1:  # multiple correspondences
                    #     obj_fp = copy.deepcopy(obj)  # necessary else the labels in objl_true get modified
                    #     obj_fp['label'] = 0  # not counted as tp (ie counted as fp), therefore label is 0
                    #     objl_pred_corresp[idx] = obj_fp
                    # elif len(indices) < 1:  # no correspondence
                    #     obj_fp = copy.deepcopy(obj)  # necessary else the labels in objl_true get modified
                    #     obj_fp['label'] = 0  # fp, therefore label is 0
                    #     objl_pred_corresp[idx] = obj_fp
                    else:
                        print('Exception!! This case should never happen.')

                # First, get objl_tp:
                pred_match_idx_list = np.unique(pred_match_idx_list)  # necessary in case 1 pred matches several true. Therefore 1 pred is counted only once as tp
                pred_match_idx_list = np.flip(np.sort(pred_match_idx_list))  # sort idx in descending order, else it is a mess when deleting elements

                # Get all predictions that have a match in objl_true (i.e., objl_tp):
                objl_tp = []
                for idx in pred_match_idx_list:
                    objl_tp.append(objl_pred[idx])

                # Get all predictions that do not have a match in objl_true:
                objl_pred_no_corresp = copy.deepcopy(objl_pred) # initialize
                for idx in pred_match_idx_list:
                    del objl_pred_no_corresp[idx]

                # All obj in objl_pred_no_corresp are false positives. So we add them in objl_true_no_corresp with label 0
                objl_true_no_corresp = copy.deepcopy(objl_pred_no_corresp)
                for idx, obj in enumerate(objl_true_no_corresp):
                    obj['label'] = 0
                #test_objl_fp1 = ol.get_class(objl_true_no_corresp, label=0)
                #test_objl_fp2 = ol.get_class(objl_true_corresp, label=0)

                test_objl_fn1 = ol.get_class(objl_pred_no_corresp, 0)
                test_objl_fn2 = ol.get_class(objl_pred_corresp, 0)

                # Finally:
                objl_true_corresp = objl_true_corresp + objl_true_no_corresp
                objl_pred_corresp = objl_pred_corresp + objl_pred_no_corresp

                # Now it is easy to get objl_fp and objl_fn:
                objl_fp = ol.get_class(objl_true_corresp, label=0)
                objl_fn = ol.get_class(objl_pred_corresp, label=0)

                # Prepare the inputs of ConfusionMatrix:
                y_true = [obj['label'] for obj in objl_true_corresp]
                y_pred = [obj['label'] for obj in objl_pred_corresp]

            else:  # if nothing has been detected
                y_true = [obj['label'] for obj in objl_true_corresp]
                y_pred = [0 for _ in range(len(y_true))]

            # Get ConfusionMatrix:
            cmat = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)

            # Store:
            self.detect_eval[tomoid]['cmat'] = cmat
            self.detect_eval[tomoid]['objl_tp'] = objl_tp
            self.detect_eval[tomoid]['objl_fp'] = objl_fp
            self.detect_eval[tomoid]['objl_fn'] = objl_fn

            y_true_global += y_true
            y_pred_global += y_pred

        cmat_global = ConfusionMatrix(actual_vector=y_true_global, predict_vector=y_pred_global)
        self.detect_eval['global']['cmat'] = cmat_global






    def get_metrics(self):
        n_true_global = 0
        n_pred_global = 0
        for tomoid in self.tomoid_keys:
            self.detect_eval[tomoid]['pre'] = self.detect_eval[tomoid]['cmat'].PPV
            self.detect_eval[tomoid]['rec'] = self.detect_eval[tomoid]['cmat'].TPR
            self.detect_eval[tomoid]['f1s'] = self.detect_eval[tomoid]['cmat'].F1
            self.detect_eval[tomoid]['n_tp'] = self.detect_eval[tomoid]['cmat'].TP
            self.detect_eval[tomoid]['n_fp'] = self.detect_eval[tomoid]['cmat'].FP
            self.detect_eval[tomoid]['n_fn'] = self.detect_eval[tomoid]['cmat'].FN

            n_true = len(self.detect_eval[tomoid]['objl_true'])
            n_pred = len(self.detect_eval[tomoid]['objl_pred'])
            self.detect_eval[tomoid]['n_true'] = n_true
            self.detect_eval[tomoid]['n_pred'] = n_pred

            n_true_global += n_true
            n_pred_global += n_pred


        self.detect_eval['global']['pre'] = self.detect_eval['global']['cmat'].PPV
        self.detect_eval['global']['rec'] = self.detect_eval['global']['cmat'].TPR
        self.detect_eval['global']['f1s'] = self.detect_eval['global']['cmat'].F1
        self.detect_eval['global']['n_tp'] = self.detect_eval['global']['cmat'].TP
        self.detect_eval['global']['n_fp'] = self.detect_eval['global']['cmat'].FP
        self.detect_eval['global']['n_fn'] = self.detect_eval['global']['cmat'].FN

        self.detect_eval['global']['n_true'] = n_true_global
        self.detect_eval['global']['n_pred'] = n_pred_global


    # def get_cm(self, dist_thr):
    #     self.get_distance_matrix()
    #
    #     z_true_global = []
    #     z_pred_global = []
    #     n_multi_hit = 0
    #     for key, dmat in self.dmat_dict.items():
    #         y_true = [obj['class'] for obj in self.dset_true[key]['object_list']]
    #         y_pred = [obj['class'] for obj in self.dset_pred[key]['object_list']]
    #
    #         if dmat is not None:  # i.e., if something has been detected
    #             # y_true and y_pred are not of same lenght. Below I transform y_pred into z_pred and y_true into z_true,
    #             # where z_true and z_pred have 1 to 1 correspondence between their elements:
    #             z_pred = [None for _ in range(len(y_true))]
    #
    #             # Correspondence matrix where '1' means that an entry from pred is situated at a distance <=dist_thr
    #             # to an entry of true:
    #             dmat = dmat <= dist_thr
    #
    #             match_idx_list = []  # this list stores all the entries in y_pred that have a match in y_true
    #             for idx in range(len(y_true)):
    #                 indices = np.nonzero(dmat[idx, :])[0]
    #                 if len(indices) == 1:  # only 1 correspondence = true positive // this is necessary in case 1 true matches several pred
    #                     z_pred[idx] = y_pred[indices[0]]
    #                     match_idx_list.append(indices[0])
    #                 elif len(indices) > 1:  # multiple correspondences (not counted as tp)
    #                     z_pred[idx] = 0
    #                     n_multi_hit += 1
    #                     print(f'multi hit: {len(indices)}')
    #                 elif len(indices) < 1:  # no correspondence = false negative
    #                     z_pred[idx] = 0
    #                 else:
    #                     print(f'Exception!! {len(indices)}')
    #
    #             # At this point, z_pred has a 1-to-1 corresp to y_true. But we also have to take into account the
    #             # predictions that have no correspondence in GT, because those are false positives. In z_true, these
    #             # correspond to the negative class (lbl=0)
    #             y_pred_no_corresp = y_pred
    #
    #             match_idx_list = np.unique(match_idx_list)  # this is necessary in case 1 pred matches several true. Therefore 1 pred is counted only once as true positive
    #             match_idx_list = np.flip(np.sort(match_idx_list))  # sort in descending order, else it is a mess deleting elements y idx
    #             for idx in match_idx_list:
    #                 del y_pred_no_corresp[idx]
    #             y_true_no_corresp = [0 for _ in range(len(y_pred_no_corresp))]  # give label 0
    #
    #             z_true = y_true + y_true_no_corresp
    #             z_pred = z_pred + y_pred_no_corresp
    #
    #         else:  # if nothing has been detected
    #             z_true = y_true
    #             z_pred = [0 for _ in range(len(z_true))]
    #
    #         z_true_global += z_true
    #         z_pred_global += z_pred
    #
    #         # print(y_found)
    #
    #     cm = ConfusionMatrix(actual_vector=z_true_global, predict_vector=z_pred_global)
    #     return cm
    #
    # def get_cm_wrt_dist(self, dist_thr_list):
    #     cm_list = []
    #     for dist_thr in dist_thr_list:
    #         cm = self.get_cm(dist_thr)
    #         cm_list.append(cm)
    #
    #     return cm_list
    #
    # def get_scores_wrt_dist(self, dist_thr_list):
    #     cm_list = self.get_cm_wrt_dist(dist_thr_list)
    #     pre_list = []
    #     rec_list = []
    #     f1s_list = []
    #     for cm in cm_list:
    #         pre_list.append(cm.PPV)
    #         rec_list.append(cm.TPR)
    #         f1s_list.append(cm.F1)
    #
    #     return pre_list, rec_list, f1s_list