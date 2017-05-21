import pickle
import os
from sklearn.preprocessing import StandardScaler


def scale_norm(x):
    # Fit a per-column scaler
    scaler = StandardScaler().fit(x)
    # Apply the scaler to X
    scaled = scaler.transform(x)
    return scaled, scaler


SVC_FILE = "svc.pkl"

def save_classifier(svc, scaler, cspace, spatial_size,
                           hist_bins, orient,
                           pix_per_cell, cell_per_block, hog_channel,
                           spatial_feat, hist_feat, hog_feat):
    try:
        of = open(SVC_FILE, 'wb')
        save = {
            'svc': svc,
            'scaler': scaler,
            'color_space': cspace,
            'orient': orient,
            'pix_per_cell': pix_per_cell,
            'cell_per_block': cell_per_block,
            'spatial_size': spatial_size,
            'hist_bins': hist_bins,
            'hog_channel': hog_channel,
            'spatial_feature': spatial_feat,
            'histogram_feature': hist_feat,
            'hog_feature': hog_feat
        }
        pickle.dump(save, of, pickle.HIGHEST_PROTOCOL)
        of.close()
    except Exception as e:
        print('Unable to save data to', SVC_FILE, ':', e)
        raise


def load_classifier():
    with open(SVC_FILE, mode='rb') as inf:
        classifier = pickle.load(inf)

    svc = classifier["svc"]
    scaler = classifier["scaler"]
    cspace = classifier["color_space"]
    orient = classifier["orient"]
    pix_per_cell = classifier["pix_per_cell"]
    cell_per_block = classifier["cell_per_block"]
    spatial_size = classifier["spatial_size"]
    hist_bins = classifier["hist_bins"]
    hog_channel = classifier["hog_channel"]
    spatial_feat = classifier["spatial_feature"]
    hist_feat = classifier["histogram_feature"]
    hog_feat = classifier["hog_feature"]
    return svc, scaler, cspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat

def have_classifier():
    return os.path.exists(SVC_FILE)
