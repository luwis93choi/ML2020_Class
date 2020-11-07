import numpy as np
import matplotlib.pyplot as plt
import sklearn
import mglearn

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

digits = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks':(), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)