import h5py, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path
import pickle

from filelock import FileLock
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from scipy.stats import spearmanr
import sh
import wandb
from wandb.keras import WandbCallback
import yaml

import model_zoo, utils # remove the from
import logomaker
import tfomics
from tfomics import impress, explain, moana

from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


def make_directory(dir_name: str):
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    return

def Spearman(y_true, y_pred):
     return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                       tf.cast(y_true, tf.float32)], Tout = tf.float32) )
from keras import backend as K

def pearson_r(y_true, y_pred):
    # use smoothing for not resulting in NaN values
    # pearson correlation coefficient
    # https://github.com/WenYanger/Keras_Metrics
    epsilon = 10e-5
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return K.mean(r)

# create functions
def summary_statistics(model, X, Y):
    pred = model.predict(X, batch_size=512)

    mse = mean_squared_error(Y[:,0], pred[:,0])
    pcc = stats.pearsonr(Y[:,0], pred[:,0])[0]
    scc = stats.spearmanr(Y[:,0], pred[:,0])[0]

    print('MSE = ' + str("{0:0.4f}".format(mse)))
    print('PCC = ' + str("{0:0.4f}".format(pcc)))
    print('SCC = ' + str("{0:0.4f}".format(scc)))

    return mse, pcc, scc

def data_splits(N, test_split, valid_split, rnd_seed): # TODO: set seed to be reproducible!
    train_split = 1 - test_split - valid_split
    shuffle = np.random.permutation(range(N))
    num_valid = int(valid_split*N)
    num_test = int(test_split*N)
    test_index = shuffle[:num_test]
    valid_index = shuffle[num_test:num_test+num_valid]
    train_index = shuffle[num_test+num_valid:]
    return train_index, valid_index, test_index

def evaluate_model(model, X, Y, task):

    i = 0 if task == "Dev" else 1

    pred = model.predict(X, batch_size=512) #[i].squeeze()
    if len(pred) == 2:
        pred = pred[i].squeeze()
    else:
        pred = pred[:, i]

    mse = mean_squared_error(Y[:, i], pred)
    pcc = stats.pearsonr(Y[:, i], pred)[0]
    scc = stats.spearmanr(Y[:, i], pred)[0]

    print(f"{task} MSE = {str('{0:0.3f}'.format(mse))}")
    print(f"{task} PCC = {str('{0:0.3f}'.format(pcc))}")
    print(f"{task} SCC = {str('{0:0.3f}'.format(scc))}")

    return mse, pcc, scc

def load_deepstarr_data(
        data_split: str,
        data_dir='/home/chandana/projects/hominid_pipeline/data/deepstarr_data.h5',
        subsample: bool = False
    ) -> (np.ndarray, np.ndarray):
    """Load dataset"""

    # load sequences and labels
    with FileLock(os.path.expanduser(f"{data_dir}.lock")):
        with h5py.File(data_dir, "r") as dataset:
            x = np.array(dataset[f'x_{data_split}']).astype(np.float32)
            y = np.array(dataset[f'y_{data_split}']).astype(np.float32).transpose()
    if subsample:
        if data_split == "train":
            x = x[:80000]
            y = y[:80000]
        elif data_split == "valid":
            x = x[:20000]
            y = y[:20000]
        else:
            x = x[:10000]
            y = y[:10000]
    return x, y

def hominid_pipeline(config):

    # ==============================================================================
    # Load dataset
    # ==============================================================================

    x_train, y_train = load_deepstarr_data("train", subsample=False)
    x_valid, y_valid = load_deepstarr_data("valid", subsample=False)
    x_test, y_test = load_deepstarr_data("test", subsample=False)

    N, L, A = x_train.shape
    output_shape = y_train.shape[-1]

    print(f"Input shape: {N, L, A}. Output shape: {output_shape}")

    config["input_shape"] = (L, A)
    config["output_shape"] = output_shape

    print(output_shape)

    # ==============================================================================
    # Build model
    # ==============================================================================

    print("Building model...")

    model = model_zoo.base_model(**config)

    return x_train, y_train, x_valid, y_valid, x_test, y_test, model

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

    position_interactions = get_position_interactions(att_maps, thresh)
    num_rands = int(random_frac/(1-random_frac))
    random_interactions = [np.random.randint(len(att_maps), size=(num_rands, 1)), np.random.randint(att_maps.shape[1], size=(num_rands, 2))]
    position_pairs = [np.vstack([position_interactions[0], random_interactions[0]]), np.vstack([position_interactions[1], random_interactions[1]])]
    if limit is not None:
        permutation = np.random.permutation(len(position_pairs[0]))
        position_pairs = [position_pairs[0][permutation], position_pairs[1][permutation]]
        position_pairs = [position_pairs[0][:limit], position_pairs[1][:limit]]

    filter_interactions = feature_maps[position_pairs].transpose([1, 2, 0])
    correlation_matrix = correlation(filter_interactions[0], filter_interactions[1])
    if symmetrize is not None:
        correlation_matrix = symmetrize(np.array([correlation_matrix, correlation_matrix.transpose()]), axis=0)
    correlation_matrix = np.nan_to_num(correlation_matrix)

    return correlation_matrix


def get_position_interactions(att_maps, threshold=0.1):
    position_interactions = np.array(np.where(att_maps >= threshold))
    position_interactions = [position_interactions[[0]].transpose(), position_interactions[[1, 2]].transpose()]
    return position_interactions


def correlation(set1, set2, function=pearsonr):
    combinations = np.indices(dimensions=(set1.shape[0], set2.shape[0])).transpose().reshape((-1, 2)).transpose()[::-1]
    vector_mesh = [set1[combinations[0]], set2[combinations[1]]]
    vector_mesh = np.array(vector_mesh).transpose([1, 0, 2])
    correlations = []
    for i in range(len(vector_mesh)):
        r = function(vector_mesh[i][0], vector_mesh[i][1])
        correlations.append(r)
    correlations = np.array(correlations).reshape((len(set1), len(set2)))
    return correlations


def filter_max_align_batch(X, model, layer=3, window=24, threshold=0.5, batch_size=1024, max_align=1e4, verbose=1):
  """get alignment of filter activations for visualization"""
  if verbose:
    print("Calculating filter PPM based on activation-based alignments")

  N,L,A = X.shape
  num_filters = model.layers[layer].output.shape[2]

  # Set the left and right window sizes
  window_left = int(window/2)
  window_right = window - window_left

  # get feature maps of 1st convolutional layer after activation
  intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)

  # batch the data
  dataset = tf.data.Dataset.from_tensor_slices(X)
  batches = dataset.batch(batch_size)

  # loop over batches to capture MAX activation
  if verbose:
    print('  Calculating MAX activation')
  MAX = np.zeros(num_filters)
  for x in batches:

    # get feature map for mini-batch
    fmap = intermediate.predict(x, verbose=0)

    # loop over each filter to find "active" positions
    for f in range(num_filters):
      MAX[f] = np.maximum(MAX[f], np.max(fmap[:,:,f]))


  # loop over each filter to find "active" positions
  W = []
  counts = []
  for f in tqdm(range(num_filters)):
    if verbose:
      print("    processing %d out of %d filters"%(f+1, num_filters))
    status = 0

    # compile sub-model to get feature map
    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output[:,:,f])

    # loop over each batch
    #dataset = tf.data.Dataset.from_tensor_slices(X)
    seq_align_sum = np.zeros((window, A)) # running sum
    counter = 0                            # counts the number of sequences in alignment
    status = 1                            # monitors whether depth of alignment has reached max_align
    for x in batches:
      if status:

        # get feature map for a batch sequences
        fmaps = intermediate.predict(x, verbose=0)

        # Find regions above threshold
        for data_index, fmap in enumerate(fmaps):
          if status:
            pos_index = np.where(fmap > MAX[f] * threshold)[0]

            # Make a sequence alignment centered about each activation (above threshold)
            for i in range(len(pos_index)):
              if status:
                # Determine position of window about each filter activation
                start_window = pos_index[i] - window_left
                end_window = pos_index[i] + window_right

                # Check to make sure positions are valid
                if (start_window > 0) & (end_window < L):
                  seq_align_sum += x[data_index,start_window:end_window,:].numpy()
                  counter += 1
                  if counter == max_align:
                    status = 0
                else:
                  break
          else:
            break
      else:
        if verbose:
          print("      alignment has reached max depth for all filters")
        break

    # calculate position probability matrix of filter
    if verbose:
      print("      %d sub-sequences above threshold"%(counter))
    if counter > 0:
      W.append(seq_align_sum/counter)
    else:
      W.append(np.ones((window,A))/A)
    counts.append(counter)
  return np.array(W), np.array(counts)


def calculate_filter_activations(
    model,
    x_test,
    params_path,
    layer,
    threshold,
    window,
    batch_size=64,
    plot_filters=False,
    from_saved=False
    ):
    """Calculate filter activations and return the final results."""
    filters_path = os.path.join(params_path, 'filters')

    if from_saved == False:
        make_directory(filters_path)

        print("Making intermediate predictions...")

        # Get the intermediate layer model
        intermediate = tf.keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)

        # Open the file to append predictions
        predictions_file = f'{filters_path}/filters_{layer}.pkl'

        # Remove the file if it already exists:
        path = Path(predictions_file)
        if path.is_file():
            sh.rm(predictions_file)

        with open(predictions_file, 'ab') as file:
            # Iterate over batches
            for batch in batch_generator(x_test, batch_size):
                # Get predictions for the batch
                batch_predictions = intermediate.predict(batch)

                # Append predictions to the pickle file
                pickle.dump(batch_predictions, file)

        # Load predictions from the pickle file
        fmap = load_predictions_from_file(predictions_file)

        # Remove predictions file:
        path = Path(predictions_file)
        if path.is_file():
            print("Now removing the predictions file!")
            sh.rm(predictions_file)

        # Concatenate the predictions into a single array
        fmap = np.concatenate(fmap, axis=0)

        # Perform further calculations on fmap
        print("Calculating filter activations...")
        W, counts = activation_pwm(fmap, x_test, threshold=threshold, window=window)

        # Filter out empty filters:
        # Check if batches have all elements not equal to 0.25
        batch_flags = np.all(W != 0.25, axis=2)

        # Find the indices of batches with all elements not equal to 0.25
        batch_indices = np.where(batch_flags.sum(axis=-1))[0]
        print(f"Learned filters : empty filters = {len(batch_indices)} : {len(W)}")

        # Select batches from the original array
        sub_W = W[batch_indices]

        # Clip filters for TomTom
        W_clipped = utils.clip_filters(sub_W, threshold=0.5, pad=3)
        moana.meme_generate(W_clipped, output_file=f"{filters_path}/filters_{layer}.txt")

        # save filter PWMs to an h5 file
        with h5py.File(f"{filters_path}/filters_{layer}.h5", "w") as f:
            dset = f.create_dataset(name="filters", data=W, dtype='float32')
            dset = f.create_dataset(name="filters_subset", data=sub_W, dtype='float32')
            dset = f.create_dataset(name="counts", data=counts, dtype='float32')

        # write jaspar file for RSAT:
        print("Writing output for RSAT...")
        output_file = f"{filters_path}/filters_{layer}_hits.jaspar"

        path = Path(output_file)
        if path.is_file():
            sh.rm(output_file)

        # get the position frequency matrix
        pfm = np.array([W[i] * counts[i] for i in range(len(counts))])

        # write jaspar file for RSAT:
        write_filters_jaspar(output_file, pfm, batch_indices)

    # Load filter PWMs, counts from an h5 file
    print("Loading filters...")
    with h5py.File(f"{filters_path}/filters_{layer}.h5", "r") as f:
        W = f["filters"][:]
        sub_W = f["filters_subset"][:]
        counts = f["filters"][:]

    # Plot filters
    if plot_filters:
        print("Plotting filters...")
        filters_fig_path = os.path.join(filters_path, f'filters_{layer}.pdf')
        plot_filters_and_return_path(W, filters_fig_path)

    return W, sub_W, counts

def plot_filters_and_return_path(W, filters_fig_path, threshold=True):
    """Plot filters and return the path to the saved figure."""
    # Check if batches have all elements not equal to 0.25
    batch_flags = np.all(W != 0.25, axis=2)

    # Find the indices of batches with all elements not equal to 0.25
    batch_indices = np.where(batch_flags.sum(axis=-1))[0]

    # Select batches from the original array:
    sub_W = W[batch_indices] if threshold else W
    indices = batch_indices if threshold else list(range(len(W)))
    # isApple = True if fruit == 'Apple' else False
    num_plot = len(sub_W)

    # Plot filters
    fig = plt.figure(figsize=(20, num_plot // 10))
    W_df = impress.plot_filters(sub_W, fig, num_cols=10, fontsize=12, names=indices)

    fig.savefig(filters_fig_path, format='pdf', dpi=200, bbox_inches='tight')

    return

def batch_generator(data, batch_size):
    """Generate batches from data."""
    num_batches = len(data) // batch_size
    for i in tqdm(range(num_batches)):
        yield data[i * batch_size : (i + 1) * batch_size]

def load_predictions_from_file(file_path):
    """Load predictions from a pickle file."""
    predictions = []
    with open(file_path, 'rb') as file:
        while True:
            try:
                batch_predictions = pickle.load(file)
                predictions.append(batch_predictions)
            except EOFError:
                break
    return predictions


def activation_pwm(fmap, X, threshold=0.5, window=20):

    # extract sequences with aligned activation
    window_left = int(window/2)
    window_right = window - window_left

    N,L,A = X.shape
    num_filters = fmap.shape[-1]

    W = []
    counts = []
    for filter_index in tqdm(range(num_filters)):
        counter = 0

        # find regions above threshold
        coords = np.where(fmap[:,:,filter_index] > np.max(fmap[:,:,filter_index])*threshold)

        if len(coords) > 1:
            x, y = coords

            # sort score
            index = np.argsort(fmap[x,y,filter_index])[::-1]
            data_index = x[index].astype(int)
            pos_index = y[index].astype(int)

            # make a sequence alignment centered about each activation (above threshold)
            seq_align = []
            for i in range(len(pos_index)):

                # determine position of window about each filter activation
                start_window = pos_index[i] - window_left
                end_window = pos_index[i] + window_right

                # check to make sure positions are valid
                if (start_window > 0) & (end_window < L):
                    seq = X[data_index[i], start_window:end_window, :]
                    seq_align.append(seq)
                    counter += 1

            # calculate position probability matrix
            if len(seq_align) > 1:#try:
                W.append(np.mean(seq_align, axis=0))
            else:
                W.append(np.ones((window,4))/4)
        else:
            W.append(np.ones((window,4))/4)
        counts.append(counter)
    return np.array(W), np.array(counts)

@tf.function
def saliency_map(X, model, class_index=0):

    if not tf.is_tensor(X):
        X = tf.Variable(X)

    with tf.GradientTape() as tape:
        tape.watch(X)
        outputs = model(X)[:, class_index]
    grad = tape.gradient(outputs, X)
    return grad

def write_filters_jaspar(output_file, pfm, batch_indices):

    # open file for writing
    f = open(output_file, 'w')
    sub_pfm = pfm[batch_indices]
    for i, pwm in enumerate(sub_pfm):

        f.write(f">filter_{batch_indices[i]} filter_{batch_indices[i]}\n")

        for j, base in enumerate("ACGT"):

            terms = [f"{value:6.2f}" for value in pwm.T[j]]
            line = f"{base} [{' '.join(terms)}]\n"

            f.write(line)

    f.close()

    return


def clip_filters(W, threshold=0.5, pad=3):
  """clip uninformative parts of conv filters"""
  W_clipped = []
  for w in W:
    L,A = w.shape
    entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
    index = np.where(entropy > threshold)[0]
    if index.any():
      start = np.maximum(np.min(index)-pad, 0)
      end = np.minimum(np.max(index)+pad+1, L)
      W_clipped.append(w[start:end,:])
    else:
      W_clipped.append(w)

  return W_clipped


def meme_generate(W, output_file='meme.txt', prefix='filter'):
  """generate a meme file for a set of filters, W ∈ (N,L,A)"""

  # background frequency
  nt_freqs = [1./4 for i in range(4)]

  # open file for writing
  f = open(output_file, 'w')

  # print intro material
  f.write('MEME version 4\n')
  f.write('\n')
  f.write('ALPHABET= ACGT\n')
  f.write('\n')
  f.write('Background letter frequencies:\n')
  f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
  f.write('\n')

  for j, pwm in enumerate(W):
    L, A = pwm.shape
    f.write('MOTIF %s%d \n' % (prefix, j))
    f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
    for i in range(L):
      f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
    f.write('\n')

  f.close()
