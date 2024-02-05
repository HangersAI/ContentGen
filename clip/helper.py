import os
import numpy as np
import sys
from PIL import Image
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import clip


class CategoryIdentifier(object):
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load(str(ROOT / "ckpts/RN50.pt"), device=device)
        self.categories = []
        self.category_encodes = None
        self.fashion_categories = []
        self.fashion_category_encodes = None

    def setup_category_encodes(self, new_categories, new_fashion_categories):
        if self.categories != new_categories:
            self.categories = new_categories
            self.category_encodes = clip.tokenize(self.categories).to(self.device)
        if self.fashion_categories != new_fashion_categories:
            self.fashion_categories = new_fashion_categories
            self.fashion_category_encodes = clip.tokenize(self.fashion_categories).to(self.device)

    def __call__(self, image_path, categories, fashion_categories):
        self.setup_category_encodes(categories, fashion_categories)
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        logits_per_image, logits_per_text = self.model(image, self.category_encodes)
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
        cls_pred = np.argmax(probs[0])
        cate_pred = self.categories[cls_pred]
        if cate_pred != "Fashion product":
            return cate_pred
            
        logits_per_image, logits_per_text = self.model(image, self.fashion_category_encodes)
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
        cls_pred = np.argmax(probs[0])
        cate_pred = self.fashion_categories[cls_pred]
        return cate_pred