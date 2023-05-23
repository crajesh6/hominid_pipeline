import time
import h5py
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import json

from multiprocess import Pool
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

from hominid_pipeline import utils



def absmaxND(a, axis=None):
    amax = np.max(a, axis)
    amin = np.min(a, axis)
    return np.where(-amin > amax, amin, amax)


def get_layer_output(model, index, X):
    temp = tf.keras.Model(inputs=model.inputs, outputs=model.layers[index].output)
    return temp.predict(X)

def pearsonr(vector1, vector2):
    m1 = np.mean(vector1)
    m2 = np.mean(vector2)

    diff1 = vector1 - m1
    diff2 = vector2 - m2

    top = np.sum(diff1 * diff2)
    bottom = np.sum(np.power(diff1, 2)) * np.sum(np.power(diff2, 2))
    bottom = np.sqrt(bottom)

    return top/bottom

def plot_glifac(ax, correlation_matrix, filter_labels, vmin=-0.5, vmax=0.5):
    ax.set_xticks(list(range(len(filter_labels))))
    ax.set_yticks(list(range(len(filter_labels))))
    ax.set_xticklabels(filter_labels, rotation=90)
    ax.set_yticklabels(filter_labels)
    c = ax.imshow(correlation_matrix, cmap='bwr_r', vmin=vmin, vmax=vmax)
    return ax, c


def correlation_matrix(model, c_index, mha_index, X, thresh=0.1, random_frac=0.5, limit=None, head_concat=np.max, symmetrize=absmaxND):

    """
    * model                  trained tensorflow model
    * c_index                index of the convolutoinal layer (after pooling)
    * mha_index              index of multi-head attention layer
    * X                      test sequences
    * thresh                 attention threshold
    * random_frac            proportion of negative positions in the set of position interactions
    * limit                  maximum number of position interactions processed; sometimes needed to avoid resource exhaustion
    * head_concat            function for concatenating heads; e.g. np.max, np.mean
    * symmetrize             function for symmetrizing the correlation matrix across diagonal
    """

    assert 0 <= random_frac < 1

    feature_maps = get_layer_output(model, c_index, X)
    o, att_maps = get_layer_output(model, mha_index, X)
    att_maps = head_concat(att_maps, axis=1)

    print(f"Shape of feature maps: {feature_maps.shape}")
    print(f"Shape of att_maps maps: {att_maps.shape}")

    position_interactions = get_position_interactions(att_maps, thresh)
    print(f"Number of position interactions: {len(position_interactions)}")
    print(position_interactions[0].shape)
    num_rands = int(random_frac/(1-random_frac))
    random_interactions = [np.random.randint(len(att_maps), size=(num_rands, 1)), np.random.randint(att_maps.shape[1], size=(num_rands, 2))]
    position_pairs = [np.vstack([position_interactions[0], random_interactions[0]]), np.vstack([position_interactions[1], random_interactions[1]])]
    if limit is not None:
        permutation = np.random.permutation(len(position_pairs[0]))
        position_pairs = [position_pairs[0][permutation], position_pairs[1][permutation]]
        position_pairs = [position_pairs[0][:limit], position_pairs[1][:limit]]
    #
    print("Created the stack of position pairs!")
    filter_interactions = feature_maps[position_pairs].transpose([1, 2, 0])

    print("Created the stack of filter interactions!")

    print("Going into the correlation function!")
    # correlation_matrix = correlation(filter_interactions[0], filter_interactions[1])
    # if symmetrize is not None:
    #     correlation_matrix = symmetrize(np.array([correlation_matrix, correlation_matrix.transpose()]), axis=0)
    # correlation_matrix = np.nan_to_num(correlation_matrix)

    # return correlation_matrix
    return filter_interactions[0], filter_interactions[1]


def get_position_interactions(att_maps, threshold=0.1):
    position_interactions = np.array(np.where(att_maps >= threshold))
    position_interactions = [position_interactions[[0]].transpose(), position_interactions[[1, 2]].transpose()]
    return position_interactions


def correlation(set1, set2, function=pearsonr):

    print("We are in the correlation function now!")
    t0 = time.time()
    combinations = np.indices(dimensions=(set1.shape[0], set2.shape[0])).transpose().reshape((-1, 2)).transpose()[::-1]
    t1 = time.time()
    print(f"Time taken to create the combinations: {t1 - t0}")
    print("Created the combinations of indices!")
    vector_mesh = [set1[combinations[0]], set2[combinations[1]]] # so this is a time heavy step
    t2 = time.time()
    print(f"Time taken to create the vector_mesh: {t2 - t1}")
    print("Created the vector mesh!")
    vector_mesh = np.array(vector_mesh).transpose([1, 0, 2])
    t3 = time.time()
    print(f"Time taken to transpose the vector_mesh: {t3 - t2}")
    print("Created transposed vector mesh!")
    correlations = []
    for i in tqdm(range(len(vector_mesh))):
        r = function(vector_mesh[i][0], vector_mesh[i][1]) # this step doesn't seem to be that bad using this truncated dset
        correlations.append(r)
    correlations = np.array(correlations).reshape((len(set1), len(set2)))
    return correlations


def correlation_scratch(set1, set2, function=pearsonr):
    correlations = np.empty((set1.shape[0], set2.shape[0]))
    for i in tqdm(range(set1.shape[0])):
        for j in range(set2.shape[0]):
            correlations[i, j] = function(set1[i], set2[j])
    return correlations


params_path = "/home/chandana/ray_results/tune_hominid_pipeline-test/tune_hominid_5c0ee_00193_193_conv1_activation=relu,conv1_attention_pool_size=10,conv1_batchnorm=False,conv1_channel_weight=se,conv_2023-05-05_06-53-12"
config = json.load(open(f"{params_path}/params.json"))
weights_path = f"{params_path}/weights"

config = json.load(open(f"{params_path}/params.json"))
_, _, _, _, x_test, y_test, model = utils.hominid_pipeline(config)

model.compile(
    tf.keras.optimizers.Adam(lr=0.001),
    loss='mse',
    metrics=[utils.Spearman, utils.pearson_r]
    )
print(model.summary())

model.load_weights(f'{params_path}/weights')
print("Finished loading the model!")

print(x_test.shape)
# GLIFAC analysis
sample = x_test[:50]
lays = [type(i) for i in model.layers]
c_index = lays.index(tf.keras.layers.MaxPool1D)
mha_index = 10 # lays.index(MultiHeadAttention2) # CHANGE THIS ACCORDINGLY!
s1, s2 = correlation_matrix(
                            model,
                            c_index,
                            mha_index,
                            sample,
                            thresh=0.1,
                            random_frac=0.3,
                            limit=150000
                        )

#############################################################################


def calculate_correlation(args):
    set1, set2, i, j = args
    # print(set1[i].shape)
    print(i)

    return pearsonr(set1[i], set2[j])

def correlation_parallel(set1, set2, function=pearsonr):
    t0 = time.time()
    pool = Pool(20)  # creates a pool of process, controls worksers
    # create iterable for all pairs of vectors and include set1 and set2
    pairs = [(set1, set2, i, j) for i in range(set1.shape[0]) for j in range(set2.shape[0])]
    # map calculate_correlation function to all pairs
    correlations = pool.map(calculate_correlation, pairs)
    pool.close()  # close the pool to prevent any more tasks from being submitted to the pool
    pool.join()  # wait for the worker processes to exit
    correlations = np.array(correlations).reshape((set1.shape[0], set2.shape[0]))
    t1 = time.time()
    print(f"Time taken: {t1 - t0}")
    return correlations

corr_d = correlation_parallel(s1, s2)
corr_d = np.nan_to_num(corr_d)

print(np.allclose(corr_a, corr_d)) # This is true :)
corr_c
