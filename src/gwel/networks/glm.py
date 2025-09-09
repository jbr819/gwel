import os
from gwel.network import Segmenter
import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm

try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn","joblib"])
    from sklearn.linear_model import LogisticRegression
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib

class GLMSegmenter(Segmenter):
    def __init__(self, model_path: str, class_names: list ):
        self.model_path = model_path
        self.load_weights(model_path)
        self.channels = class_names

    def set_device(self, device: str):
        pass

    def load_weights(self, weights: str = None):
        path = weights or self.model_path
        self.model = joblib.load(path)

    def inference(self, image: np.ndarray):

        H, W, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3)

        y_pred = self.model.predict(pixels)          # shape [H*W]
       # y_proba = self.model.predict_proba(pixels)   # shape [H*W, C]

        results = {}
        for i, class_name in enumerate(self.channels):
            mask = (y_pred == i).astype(np.uint8).reshape(H, W)
            rle = mask_utils.encode(np.asfortranarray(mask))
            results[class_name] = rle

        return results



def load_annotations(coco_json_path):
    return COCO(coco_json_path)

def load_mask(coco, ann_ids, height, width):
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        rle = mask_utils.frPyObjects(ann['segmentation'], height, width)
        rle = mask_utils.merge(rle)
        m = mask_utils.decode(rle)
        mask = np.logical_or(mask, m)
    return mask.astype(np.uint8)

def sample_pixels(image, mask, num_samples_per_class=500):
    """
    image: HxWx3 RGB
    mask:  HxW binary (1 for leaf, 0 for background)
    """
    H, W = mask.shape
    coords_leaf = np.argwhere(mask == 1)
    coords_bg   = np.argwhere(mask == 0)

    if len(coords_leaf) == 0 or len(coords_bg) == 0:
        return [], []

    coords_leaf = coords_leaf[np.random.choice(len(coords_leaf), min(num_samples_per_class, len(coords_leaf)), replace=False)]
    coords_bg   = coords_bg[np.random.choice(len(coords_bg), min(num_samples_per_class, len(coords_bg)), replace=False)]

    samples = np.vstack([coords_leaf, coords_bg])
    labels  = np.array([1] * len(coords_leaf) + [0] * len(coords_bg))

    rgb = image[samples[:,0], samples[:,1], :]
    return rgb, labels

def train_GLM_on_coco(coco_json, image_dir, output_model_path, samples_per_image=1000):
    coco = load_annotations(coco_json)
    cat_ids = coco.getCatIds(catNms=['leaf'])
    img_ids = coco.getImgIds()

    X_all = []
    y_all = []

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids)
        if not ann_ids:
            continue

        image_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = load_mask(coco, ann_ids, img_info['height'], img_info['width'])

        X, y = sample_pixels(image, mask, num_samples_per_class=samples_per_image // 2)
        if len(X) == 0:
            continue

        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print(f"Total samples: {len(X_all)}")

    # Define kernelized softmax model
    model = Pipeline([
        ('scale', StandardScaler()),
        ('rbf', RBFSampler(gamma=0.5, n_components=10)),
        ('logreg', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500))
    ])

    model.fit(X_all, y_all)
    joblib.dump(model, output_model_path)
    print(f"✅ Model saved to: {output_model_path}")

def train_GLM_on_palettes(classes = [], image_paths = [], output_model_path, samples_per_image=1000):

    X_all = []
    y_all = []

    if len(classes) != len(image_paths):
        

    for image in image_paths:
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids)
        if not ann_ids:
            continue

        image_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = load_mask(coco, ann_ids, img_info['height'], img_info['width'])

        X, y = sample_pixels(image, mask, num_samples_per_class=samples_per_image // 2)
        if len(X) == 0:
            continue

        X_all.append(X)
        y_all.append(y)


    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.ones_like(image)

    X, y = sample_pixels(image, mask, num_samples_per_class=samples_per_image)
    if len(X) == 0:
        continue

    X_all.append(X)
    y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print(f"Total samples: {len(X_all)}")

    # Define kernelized softmax model
    model = Pipeline([
        ('scale', StandardScaler()),
        ('rbf', RBFSampler(gamma=0.5, n_components=10)),
        ('logreg', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500))
    ])

    model.fit(X_all, y_all)
    joblib.dump(model, output_model_path)
    print(f" Model saved to: {output_model_path}")
