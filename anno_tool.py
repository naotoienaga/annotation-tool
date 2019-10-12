'''
The semi-automatic annotation tool in ELAN format.
'''
import glob, json, configparser, codecs
import numpy as np
import cv2 as cv
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

my_round = lambda x:(x*2+1)//2

def read_eaf(elan_path, num_frm, fps):
    '''
    Return manual annotation from eaf file
    0:rest, 1:gesture, 2:no annotation
    '''
    # Open file
    eaf_file = open(elan_path)

    times = {}
    annotation = np.ones(num_frm, dtype=np.int32)*2

    for idx, line in enumerate(eaf_file):
        # Make dictionary
        if 'TIME_SLOT_ID=\"ts' in line:
            times[line.split('"')[1]] = line.split('"')[3]

        # Make label array
        elif '<ANNOTATION_VALUE>gesture' in line or '<ANNOTATION_VALUE>rest' in line:
            start = prev_line.split('TIME_SLOT_REF1=\"')[1].split('"')[0]
            start = int(np.round(fps * int(times[start]) / 1000))
            end = prev_line.split('TIME_SLOT_REF2=\"')[1].split('"')[0]
            end = int(np.round(fps * int(times[end]) / 1000))
            if 'gesture' in line:
                annotation[start:end] = 1
            elif 'rest' in line:
                annotation[start:end] = 0

        prev_line = line
    
    eaf_file.close()   
    return annotation

def read_keypoints(json_paths, th_opconf):
    '''
    Return keypoints with interpolation and normalization
    '''
    keypoints = np.full((len(json_paths), 70, 2), np.nan, dtype=np.float32)
    pose_idx_dict = {2:0, 3:1, 4:2, 5:24, 6:25, 7:26, 1:69}

    for frm_idx, json_path in enumerate(json_paths):
        # Open json
        json_data = json.load(open(json_path))
        if not json_data['people']:
            continue

        # Get each keypoints
        pose_kpts = json_data['people'][0]['pose_keypoints_2d']
        rhand_kpts = json_data['people'][0]['hand_right_keypoints_2d']
        lhand_kpts = json_data['people'][0]['hand_left_keypoints_2d']
        face_kpts = json_data['people'][0]['face_keypoints_2d']

        # load the keypoint data which has more than the predifined confidence value
        # wrists, elbows, shoulders
        for pose_key, pose_val in pose_idx_dict.items():
            keypoints[frm_idx, pose_val, :] = pose_kpts[pose_key*3], pose_kpts[pose_key*3+1] if pose_kpts[pose_key*3+2] >= th_opconf else np.nan
        
        # Hands
        for hand_idx in range(21):
            keypoints[frm_idx, hand_idx+3, :] = rhand_kpts[hand_idx*3], rhand_kpts[hand_idx*3+1] if rhand_kpts[hand_idx*3+2] >= th_opconf else np.nan
            keypoints[frm_idx, hand_idx+27, :] = lhand_kpts[hand_idx*3], lhand_kpts[hand_idx*3+1] if lhand_kpts[hand_idx*3+2] >= th_opconf else np.nan
        
        # Face
        face_idx_list = [x for x in range(3, 14)]+[x for x in range(49, 60) if x != 54]
        for kpt_idx, face_idx in enumerate(face_idx_list):
            keypoints[frm_idx, kpt_idx+48, :] = face_kpts[face_idx*3], face_kpts[face_idx*3+1] if face_kpts[face_idx*3+2] >= th_opconf else np.nan
    
    # Interpolation
    for kpt_idx in range(keypoints.shape[1]):
        df = pd.DataFrame(keypoints[:, kpt_idx, :])
        df.interpolate('linear', limit_direction='both', inplace=True)
        keypoints[:, kpt_idx, :] = df.values

    # Normalization
    for frm_idx in range(keypoints.shape[0]):
        s_length = np.linalg.norm(keypoints[frm_idx, 0, :] - keypoints[frm_idx, 24, :])
        keypoints[frm_idx] = (keypoints[frm_idx] - keypoints[frm_idx, 69, :]) / s_length

    # Replace np.nan with 0
    if np.any(np.isnan(keypoints)):
        keypoints = np.where(np.isnan(keypoints), 0., keypoints)

    return np.delete(keypoints, [69], 1)

def make_feature(keypoints, window_size, displacement):
    '''
    Return feature array
    '''
    width = int(window_size / displacement)
    x_data = []
    x_data_ap = x_data.append

    for frm_idx, keypoint in enumerate(keypoints):
        # Distance and position
        dsts = np.full((keypoints.shape[1], width*4), np.nan, dtype=np.float32)
        poss = np.full((keypoints.shape[1], width*4+2), np.nan, dtype=np.float32)

        # Calc. start and end
        start = min(width, int(frm_idx / displacement))
        end = min(width, int((keypoints.shape[0]-1 - frm_idx) / displacement))
        if start < width:
            end = 2 * width - start
        elif end < width:
            start = 2 * width - end

        # Store keypoint features
        offset = 0
        for idx, frm_idx2 in enumerate(range(-start, end+1)):
            if frm_idx2 != 0:
                dsts[:, idx*2+offset : idx*2+2+offset] = keypoint - keypoints[frm_idx+frm_idx2*displacement]
                poss[:, idx*2 : idx*2+2] = keypoints[frm_idx+frm_idx2*displacement]
            else:
                poss[:, idx*2 : idx*2+2] = keypoint
                offset -= 2

        x_data_ap(np.hstack((np.ravel(dsts), np.ravel(poss))))

    return np.array(x_data)

def cal_weight(y_data):
    '''
    Return weights of each class for training
    '''
    weight = [np.count_nonzero(y_data==1), np.count_nonzero(y_data==0)]
    return list(map(lambda x: x/max(weight), weight))

def active_learning(feature, annotation, query_frm, n_splits, epochs, early_stopping_rounds):
    '''
    Do active learning
    '''
    # Get train, validation test data
    x_data = feature[annotation != 2]
    y_data = annotation[annotation != 2]
    x_test = feature[annotation == 2]
    y_test = annotation[annotation == 2]

    # Divide x/y_data into train and validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for train_idx, valid_idx in skf.split(x_data, y_data):
        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_valid = x_data[valid_idx]
        y_valid = y_data[valid_idx]
        break

    # LightGBM parameters
    lgbm_params = {'objective':'binary', 'verbosity':-1, 'weight_column':cal_weight(y_train)}

    # Train
    lgb_train = lgb.Dataset(x_train, label=y_train)
    lgb_valid = lgb.Dataset(x_valid, label=y_valid, reference=lgb_train)
    model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_valid, num_boost_round=epochs, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

    # Test
    y_pred_prob = model.predict(x_test, num_iteration=model.best_iteration)

    # Divide unannotated frames
    unannotated = np.where(annotation == 2)[0]
    units = [x for x in range(unannotated.shape[0]-1) if unannotated[x+1]-unannotated[x] > 1]
    unannotated_divied = [(unannotated[0], unannotated[units[0]])]
    for unit_idx in range(1, len(units)):
        unannotated_divied.append((unannotated[units[unit_idx-1]+1], unannotated[units[unit_idx]]))
    unannotated_divied.append((unannotated[units[-1]+1], unannotated[-1]))

    # Select query
    tmp_arr = np.ones(annotation.shape[0])*2
    tmp_arr[annotation == 2] = y_pred_prob
    min_prob = 1.1

    for ud_idx, ud in enumerate(unannotated_divied):
        for idx in range(ud[0], ud[1]):
            end = min(ud[0] + query_frm, ud[1])

            # Calc. minimum probability
            this_prob = tmp_arr[ud[0] : end+1]
            max_prob = [this_prob[i] if this_prob[i] > 0.5 else 1-this_prob[i] for i in range(this_prob.shape[0])]
            max_prob.sort()
            this_min_prob = np.average(max_prob[:int(len(max_prob)*0.05)+1])

            # Update minimum probability
            if this_min_prob < min_prob:
                min_prob = this_min_prob
                most_uncertain = [ud[0], end+1]

            if end == ud[1]:
                break

    y_pred = np.array([int(my_round(x)) for x in y_pred_prob])
    return y_pred, most_uncertain

def organize_annotation(annotation, delete_frm, separated=None):
    '''
    Organize predicted annotation to output elan file
    '''
    if separated is None:
        # Separate annotation
        separated = []
        anno = annotation[0]
        st = 0
        for idx in range(1, annotation.shape[0]):
            if annotation[idx] != anno:
                separated.append([st, idx-1, annotation[st]])
                anno = annotation[idx]
                st = idx
        separated.append([st, idx, annotation[st]])

    if delete_frm < 0:
        return separated, True

    # Delete short annotations
    abandonment, idx, st = 0, 0, -1
    while True:
        if len(separated) <= idx:
            break

        # Change short annotations
        if separated[idx][1]-separated[idx][0] <= delete_frm and st < 0 and separated[idx][2] <= 1:
            st = idx
        elif (separated[idx][1]-separated[idx][0] > delete_frm or separated[idx][2] > 1) and st >= 0:
            if st == idx - 1:
                # Isolated short annotation is just switched to the other annotation
                separated[st][2] = 1 - separated[st][2]
            else:
                # Consecutive short annotations
                orig_st = st
                orig_ed = ed = idx - 1
                while True:
                    # Take more annotations
                    num_0 = np.count_nonzero(annotation[separated[st][0]:separated[ed][1]] == 0) + np.count_nonzero(annotation[separated[st][0]:separated[ed][1]] == 3)
                    num_1 = np.count_nonzero(annotation[separated[st][0]:separated[ed][1]] == 1) + np.count_nonzero(annotation[separated[st][0]:separated[ed][1]] == 4)
                    if num_0 != num_1:
                        label = 0 if num_0 > num_1 else 1
                        for idx2 in range(orig_st, orig_ed + 1):
                            separated[idx2][2] = label
                        break

                    # Update st and ed
                    next_st = st - 1 if st != 0 else st
                    next_ed = ed + 1 if ed < len(separated)-1 else ed
                    if next_st == st and next_ed == ed: # Give up deleting this short annotation
                        abandonment += 1
                        break
                    else:
                        st, ed = next_st, next_ed
            st = -1
        idx += 1

    # Integrate annotations
    idx = 0
    while True:
        if len(separated) <= idx + 1:
            break
        if separated[idx][2] == separated[idx+1][2]:
            separated[idx][1] = separated[idx+1][1]
            del separated[idx+1]
        else:
            idx += 1
    
    # Check if there are more short annotations that can be deleted
    num_short = 0
    for sep in separated:
        if sep[1]-sep[0] <= delete_frm:
            num_short += 1
    is_ok = True if num_short <= abandonment else False
    
    return separated, is_ok

def output_eaf(elan_path, separated, most_uncertain, fps):
    '''
    Output elan file with predicted annotation and query
    '''
    # Open file
    input_eaf = open(elan_path)
    output_eaf = open(elan_path.split('.eaf')[0]+'_predicted.eaf', 'w')

    prev_line = ''
    manual_annos = []
    times = {}
    for line in input_eaf:
        if 'TIME_SLOT_ID=\"ts' in line:
            last_ts = int(line.split('"')[1].split('s')[-1])
            times[line.split('"')[1]] = int(line.split('"')[3])
            eg_time = line

        elif '</TIME_ORDER>' in line:
            # Write TIME_SLOT
            splits = eg_time.split('"')
            for sp_idx, sp in enumerate(separated):
                for sp_idx2 in range(2):
                    ms = int(sp[sp_idx2] * 1000 / fps)
                    if sp[2] > 1 or (sp_idx != 0 and separated[sp_idx-1][2] > 1):
                        # Do not add new time slot if adjacent to manual annotation
                        separated[sp_idx][sp_idx2] = list(times.keys())[np.argmin(abs(np.array(list(times.values())) - ms))]
                    else:
                        last_ts += 1
                        separated[sp_idx][sp_idx2] = 'ts'+str(last_ts)
                        output_eaf.write('"'.join([splits[0], 'ts'+str(last_ts), splits[2], str(ms), splits[4]]))
                    if sp_idx != len(separated)-1:
                        break

            # TIME_SLOT for query
            last_ts += 1
            ms1 = int((most_uncertain[0]) * 1000 / fps)
            ms2 = int((most_uncertain[1]) * 1000 / fps)
            output_eaf.write('"'.join([splits[0], 'ts'+str(last_ts), splits[2], str(ms1), splits[4]]))
            output_eaf.write('"'.join([splits[0], 'ts'+str(last_ts+1), splits[2], str(ms2), splits[4]]))
            most_uncertain[:] = 'ts'+str(last_ts), 'ts'+str(last_ts+1)

        elif 'TIER_ID=\"' in line:
            eg_tier = line
        
        elif '<ANNOTATION_VALUE>rest' in line or '<ANNOTATION_VALUE>gesture' in line:
            last_a = int(prev_line.split('"')[1].split('a')[-1])
            manual_annos.append(prev_line)
            manual_annos.append(line)

        elif '</TIER>' in prev_line and 'TIER_ID=\"' not in line:
            tab = eg_tier.split('<')[0]
            output_eaf.write(eg_tier.split('TIER_ID="')[0] + 'TIER_ID="PREDICTED">\n')

            # Copy manual annotation to PREDICTED tier
            for idx in range(0, len(manual_annos), 2):
                last_a += 1
                output_eaf.write(tab + tab + '<ANNOTATION>\n')
                this_a = manual_annos[idx].split('"')[1]
                output_eaf.write(manual_annos[idx].replace(this_a, 'a'+str(last_a)))
                output_eaf.write(manual_annos[idx+1])
                output_eaf.write(tab + tab + tab + '</ALIGNABLE_ANNOTATION>\n')
                output_eaf.write(tab + tab + '</ANNOTATION>\n')

            # Write predicted ANNOTATION
            splits = manual_annos[0].split('"')
            r_ann = 'rest' if 'rest' in manual_annos[1] else 'gesture'
            for idx, sp in enumerate(separated):
                if sp[2] > 1: # Manual annotation
                    continue
                last_a += 1
                output_eaf.write(tab + tab + '<ANNOTATION>\n')
                if idx != len(separated)-1:
                    output_eaf.write('"'.join([splits[0], 'a'+str(last_a), splits[2], sp[0], splits[4], separated[idx+1][0], splits[6]]))
                else:
                    output_eaf.write('"'.join([splits[0], 'a'+str(last_a), splits[2], sp[0], splits[4], sp[1], splits[6]]))
                ann = 'rest' if sp[2] == 0 else 'gesture'
                output_eaf.write(manual_annos[1].replace(r_ann, ann))
                output_eaf.write(tab + tab + tab + '</ALIGNABLE_ANNOTATION>\n')
                output_eaf.write(tab + tab + '</ANNOTATION>\n')
            output_eaf.write(tab + '</TIER>\n')

            # Write query
            output_eaf.write(eg_tier.split('TIER_ID="')[0] + 'TIER_ID="QUERY">\n')
            output_eaf.write(tab + tab + '<ANNOTATION>\n')
            output_eaf.write('"'.join([splits[0], 'a'+str(last_a+1), splits[2], most_uncertain[0], splits[4], most_uncertain[1], splits[6]]))
            output_eaf.write(tab + tab + tab + tab + '<ANNOTATION_VALUE>query</ANNOTATION_VALUE>\n')
            output_eaf.write(tab + tab + tab + '</ALIGNABLE_ANNOTATION>\n')
            output_eaf.write(tab + tab + '</ANNOTATION>\n')
            output_eaf.write(tab + '</TIER>\n')
            
        prev_line = line
        output_eaf.write(line)

    input_eaf.close()
    output_eaf.close()

if __name__ == '__main__':
    # Config parser
    config = configparser.SafeConfigParser()
    config.readfp(codecs.open('config.ini', 'r', 'utf8'))

    # Config values
    fps = float(config['mandatory']['fps'])
    window_size = int(int(config['advanced']['window_size']) * fps)
    query_frm = int(int(config['mandatory']['query_sec']) * fps)
    delete_frm = int(float(config['mandatory']['delete_sec']) * fps)

    print("Reading manual annotation file...")
    # Get json paths
    json_paths = sorted(glob.glob(config['mandatory']['path_openpose_json']+'*.json'))
    # Get manual annotation
    annotation = read_eaf(config['mandatory']['path_elan'], len(json_paths), fps)

    print("Making feature array to train...")
    # Get keypoints
    keypoints = read_keypoints(json_paths, float(config['advanced']['openpose_conf']))
    # Get feature array
    feature = make_feature(keypoints, window_size, int(config['advanced']['displacement']))

    # Get predicted annotation and query
    print("Training and predicting...")
    y_pred, most_uncertain = active_learning(feature, annotation, query_frm, int(config['advanced']['n_splits']), int(config['advanced']['epochs']), int(config['advanced']['early_stopping_rounds']))    
    
    print("Outputing elan file...")
    # Organize predicted annotation
    annotation[annotation != 2] += 3 # (0, 1) -> (3, 4)
    annotation[annotation == 2] = y_pred
    separated = None
    while True:
        separated, is_ok = organize_annotation(annotation, delete_frm, separated)
        if is_ok:
            break
    # Output ELAN annotation file
    output_eaf(config['mandatory']['path_elan'], separated, most_uncertain, fps)

    print('Done! {} has been outputted.'.format(config['mandatory']['path_elan'].split('.eaf')[0]+'_predicted.eaf'))
