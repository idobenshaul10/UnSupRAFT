import pickle
import numpy as np
import sklearn.datasets
from sklearn.preprocessing import RobustScaler
import pandas as pd
import umap
import matplotlib.pyplot as plt
import sys
import os

def create_umap(X, Y, output_folder=None, layer_str="", save_graph=False, fig=None):

	print(f"Fitting umap")
	reducer = umap.UMAP(random_state=42)
		
	embedding_train = reducer.fit_transform(X)
	print(f"Done fitting umap")
	
	if fig is None:
		fig = plt.figure(figsize=(20, 16))

	
	fig, ax = plt.subplots()
	color_mapping = [int(k) for k in Y]	
	scatter = ax.scatter(
		embedding_train[:, 0], embedding_train[:, 1], c=color_mapping, cmap="inferno" , s=10
	)

	legend1 = ax.legend(*scatter.legend_elements(),
			loc="lower left", title="Classes")

	plt.setp(ax, xticks=[], yticks=[])  
	plt.suptitle(f"UMAP of RAFT features", fontsize=18)
	ax.set_xlabel(f"UMAP of features")		

	plt.show()

	# if save_graph and output_folder is not None:
	# 	save_path = os.path.join(output_folder, f"{layer_str}.png")
	# 	print(f"save_path:{save_path}")
	# 	plt.savefig(save_path, \
	# 		dpi=300, bbox_inches='tight')
	# fig.clf()

path = sys.argv[1]
features = pickle.load(open(path, "rb"))
features = np.array(features)[:100, :]

features = features.reshape(features.shape[0], -1)
create_umap(features, np.arange(len(features)), \
		output_folder='umaps', layer_str='all', save_graph=False)