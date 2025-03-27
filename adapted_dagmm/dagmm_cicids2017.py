# %% [markdown]
# # Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection

# %%
import numpy as np 
from data_loader import *
from main import *
from IPython.utils import io

# %% [markdown]
# ## CIC IDS2017

# %%
class hyperparams():
    def __init__(self, config):
        self.__dict__.update(**config)
defaults = {
    'lr' : 1e-4,
    'num_epochs' : 200,
    'batch_size' : 1024,
    'gmm_k' : 4,
    'lambda_energy' : 0.1,
    'lambda_cov_diag' : 0.005,
    'pretrained_model' : None,
    'mode' : 'train',
    'use_tensorboard' : False,
    'data_path' : '../data',
    'ds': 'cicids2017',

    'log_path' : './dagmm/logs',
    'model_save_path' : './dagmm/models',
    'sample_path' : './dagmm/samples',
    'test_sample_path' : './dagmm/test_samples',
    'result_path' : './dagmm/results',

    'log_step' : None,
    'sample_step' : 194,
    'model_save_step' : 194,
}

# %%
with io.capture_output() as captured:
    solver = main(hyperparams(defaults))
    solver.test();

# %%
label_names = ['BENIGN','FTP-Patator','SSH-Patator','DoS slowloris','DoS Slowhttptest','DoS Hulk','DoS GoldenEye',
          'Heartbleed','Web Attack','Infiltration', 'Bot', 'PortScan', 'DDoS']

# %%
y_test = solver.data_loader.dataset.real_test_labels
all_pred = solver.pred

print(4*"=", "RESULTADOS", 4*"=")

for i in range(len(label_names)):
    #### Exclude web attacks from results
    if label_names[i].find('Web')>=0:
        continue
    preds = all_pred[y_test==label_names[i]]
    if i==0:
        fpr = "{0:0.4f}".format(np.sum(preds)/(0. + len(preds)))
        print('FPR:',fpr)
    else:
        tpr = "{0:0.4f}".format(np.sum(preds)/(0. + len(preds)))
        print(label_names[i]+':',tpr)


