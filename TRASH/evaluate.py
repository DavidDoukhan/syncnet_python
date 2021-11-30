from collections import OrderedDict
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from SyncNetInstance import *

# ADDED by ITH
def evaluate(model, test_gen, test_ann_dir=None, not_speaking_label=0, speaking_label=1) -> OrderedDict:

    result = OrderedDict()
    # Evaluate
    #y_audio_pred, y_video_pred, y_main_pred = model.predict(test_gen.dataset, verbose=1)
    n=3
    test_dataset = test_gen.dataset.take(n)
    y_audio_pred, y_video_pred, y_main_pred = model.predict(test_dataset, verbose=1) # ITH

    # Make sure that the length of the result matches the annotations
    #assert len(test_gen.anns_selected) == len(y_main_pred), f"len(test_gen.anns_selected) != len(y_main_pred)"

    # Get ground truth
    # Invert labels so that speaking is 1 and not-speaking is 0
    y_true = test_gen.targets(invert=True)
    result['y_true'] = y_true

    #####################################################
    # Make classification report and confusion matrices
    #####################################################

    # Use argmin to return integer class_id values
    #
    # The model defines: speaking as 0 and not-speaking as 1
    # The ground truth defines: speaking as 1 and not-speaking as 0
    #
    # Model Speaking Example:
    # Using argmin will choose the index 1, which means speaking in the ground truth
    #   0    1   (Indexes)
    # [0.9 0.1]  (Values)
    #
    # Model Non-Speaking Example
    # Using argmin will choose the index 0, which means non-speaking in the ground truth
    #   0    1   (Indexes)
    # [0.1 0.9]  (Values)

    y_audio_class_ids = np.argmin(y_audio_pred, axis=1)
    y_video_class_ids = np.argmin(y_video_pred, axis=1)
    y_main_class_ids = np.argmin(y_main_pred, axis=1)

    result['y_audio_class_ids'] = y_audio_class_ids
    result['y_video_class_ids'] = y_video_class_ids
    result['y_main_class_ids'] = y_main_class_ids

    #####################################
    # Calculate Accuracy
    #####################################
    #s= accuracy_score(y_true, y_audio_class_ids)
    #result['audio_accuracy'] = accuracy_score(y_true, y_audio_class_ids)
    #result['video_accuracy'] = accuracy_score(y_true, y_video_class_ids)
    #result['main_accuracy'] = accuracy_score(y_true, y_main_class_ids)
    result['audio_accuracy'] = accuracy_score(y_true[0:n*16], y_audio_class_ids)
    result['video_accuracy'] = accuracy_score(y_true[0:n*16], y_video_class_ids)
    result['main_accuracy'] = accuracy_score(y_true[0:n*16], y_main_class_ids)

    #####################################
    # Calculate ActivityNet mAP results
    #####################################

    # Change to axis=1 if speaking and non-speaking labels are switched in config.yaml
    y_audio_scores = get_scores(y_audio_pred, axis=0)
    y_video_scores = get_scores(y_video_pred, axis=0)
    y_main_scores = get_scores(y_main_pred, axis=0)

    annotations = test_gen.anns_selected
    y_audio_map = calc_map_activity_net(annotations, y_audio_scores)
    y_video_map = calc_map_activity_net(annotations, y_video_scores)
    y_main_map = calc_map_activity_net(annotations, y_main_scores)

    result['audio_map'] = y_audio_map
    result['video_map'] = y_video_map
    result['main_map'] = y_main_map

    #####################################
    # Calculate ActivityNet mAP results
    # on original annotations
    #####################################

    audio_lookup = create_annotation_lookup(annotations, y_audio_scores)
    video_lookup = create_annotation_lookup(annotations, y_video_scores)
    main_lookup = create_annotation_lookup(annotations, y_main_scores)

    # Load original test annotations
    orig_annotations = load_orig_annotations(test_ann_dir)
    y_audio_omap = calc_map_activity_net_orig(audio_lookup, orig_annotations)
    y_video_omap = calc_map_activity_net_orig(video_lookup, orig_annotations)
    y_main_omap = calc_map_activity_net_orig(main_lookup, orig_annotations)

    result['orig_audio_map'] = y_audio_omap
    result['orig_video_map'] = y_video_omap
    result['orig_main_map'] = y_main_omap

    #####################################
    # Calculate scikit-learn AP
    #####################################

    y_audio_ap_sp = average_precision_score(y_true, y_audio_scores, pos_label=speaking_label)
    y_video_ap_sp = average_precision_score(y_true, y_video_scores, pos_label=speaking_label)
    y_main_ap_sp = average_precision_score(y_true, y_main_scores, pos_label=speaking_label)

    result['audio_ap_sp'] = y_audio_ap_sp
    result['video_ap_sp'] = y_video_ap_sp
    result['main_ap_sp'] = y_main_ap_sp

    y_audio_ap_ns = average_precision_score(y_true, y_audio_scores, pos_label=not_speaking_label)
    y_video_ap_ns = average_precision_score(y_true, y_video_scores, pos_label=not_speaking_label)
    y_main_ap_ns = average_precision_score(y_true, y_main_scores, pos_label=not_speaking_label)

    result['audio_ap_ns'] = y_audio_ap_ns
    result['video_ap_ns'] = y_video_ap_ns
    result['main_ap_ns'] = y_main_ap_ns

    #####################################
    # Calculate scikit-learn AUC
    #####################################
    y_audio_auc = roc_auc_score(y_true, y_audio_scores)
    y_video_auc = roc_auc_score(y_true, y_video_scores)
    y_main_auc = roc_auc_score(y_true, y_main_scores)

    result['audio_auc'] = y_audio_auc
    result['video_auc'] = y_video_auc
    result['main_auc'] = y_main_auc
    return result

def get_scores(y_pred, axis=0):

    scores = []
    for y in y_pred:
        scores.append(y[axis])
    return np.array(scores)

def calc_map_activity_net(annotations, y_pred_scores):

    ground_truth, predictions = make_groundtruth_and_predictions(annotations, y_pred_scores)
    merged = merge_groundtruth_and_predictions(ground_truth, predictions)
    precision, recall = calculate_precision_recall(merged)
    return compute_average_precision(precision, recall)

def make_groundtruth_and_predictions(annotations, y_pred_scores):
    ground_truth = []
    predictions = []

    for ann_window, score in zip(annotations, y_pred_scores):
        # Get last annotation in window, which is what we are predicting for
        ann = ann_window[-1]

        # Make ground truth row
        x1, y1, x2, y2 = ann.bbox
        g_row = [ann.vid_id, ann.timestamp, x1, y1, x2, y2, ann.label, ann.face_id]

        # Make prediction row
        p_row = [ann.vid_id, ann.timestamp, x1, y1, x2, y2, 'SPEAKING_AUDIBLE', ann.face_id, score]

        # Append rows to lists
        ground_truth.append(g_row)
        predictions.append(p_row)

    # Make ground truth and prediction data frames
    ground_truth_cols = ['video_id', 'frame_timestamp', 'entity_box_x1', 'entity_box_y1', 'entity_box_x2',
                         'entity_box_y2', 'label', 'entity_id']
    prediction_cols = ground_truth_cols + ['score']
    df_ground_truth = pd.DataFrame(data=ground_truth, columns=ground_truth_cols)
    df_predictions = pd.DataFrame(data=predictions, columns=prediction_cols)

    make_uids(df_ground_truth)
    make_uids(df_predictions)

    return df_ground_truth, df_predictions

if __name__ == "__main__":
    syncnet = SyncNetInstance();
    syncnet.loadParameters('data/syncnet_v2.model');
    evaluate(syncnet)