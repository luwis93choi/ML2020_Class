import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

# 그래프 결과를 저장한 경로 정의
PROJECT_ROOT_DIR = '.'
CHAPTER_ID = 'dim_reduction'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# 그래프 결과를 저장하는 함수
def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Save Image ', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

####################################################################################################################################################
### Prepare 3D Swiss Roll Dataset ##################################################################################################################
####################################################################################################################################################
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)	# 1000개의 데이터로 구성된 3D Swiss Roll 데이터셋을 준비함

####################################################################################################################################################
### LLE (Locally Linear Embedding) - NonLinear Dimensionality Reduction Technique / Manifold Learning ##############################################
####################################################################################################################################################

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)       
# 각 데이터 주변 10개의 Neighbor를 이용하여 2개의 최상 Principle Component를 만드는 LLE PCA 객체를 준비함

X_reduced = lle.fit_transform(X)    # Principle Component를 이용하여 데이터셋을 재구성함

# LLE로 재구성된 데이터셋(펼쳐진 Swiss Roll)을 그래프로 그림
plt.title("Unrolled swiss roll using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)

save_fig("lle_unrolling_plot")
plt.show()



