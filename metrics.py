# Necessary Packages
import tensorflow as tf
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
from utils import train_test_divide, batch_generator

def compute_discriminative_score (ori_data, generated_data):
  """Use post-hoc RNN to classify original data and synthetic data
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    
  Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
      result in range [0, 0.5], the lower the better
  """
  # -------------------------- #
  # ----- Initialization ----- #
  # -------------------------- #
  tf.compat.v1.reset_default_graph()
  tf.compat.v1.disable_eager_execution()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape    
    
  # Set maximum sequence length and each sequence length (change in case of variable length sequences)
  ori_time = np.full(shape=no, fill_value=seq_len, dtype=np.int)
  ori_max_seq_len = seq_len
  generated_time = np.full(shape=no, fill_value=seq_len, dtype=np.int)
  generated_max_seq_len = seq_len
  max_seq_len = seq_len 
     
  # ------------------------------------------------- #
  # ----- RNN Discriminative network definition ----- #
  # ------------------------------------------------- #
  # Network parameters
  hidden_dim = 4
  iterations = 2000
  batch_size = 128
    
  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  X_hat = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_hat")
    
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")
  T_hat = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t_hat")
    
  # build the discriminator
  def discriminator (x, t):
    """Discriminator network.
    
    Args:
      - x: time-series data
      - t: time information
      
    Returns:
      - y_hat_logit: logits of the discriminator output
      - y_hat: discriminator output
      - d_vars: discriminator variables
    """
    with tf.compat.v1.variable_scope("discriminator", reuse = tf.compat.v1.AUTO_REUSE) as vs:
      d_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
      d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length = t)
      y_hat_logit = tf.compat.v1.layers.dense(d_last_states, 1, activation=None) 
      y_hat = tf.nn.sigmoid(y_hat_logit)
      d_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(vs.name)]
    
    return y_hat_logit, y_hat, d_vars
    
  y_logit_real, y_pred_real, d_vars = discriminator(X, T)
  y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)
        
  # Loss for the discriminator
  d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_real, 
                                                                      labels = tf.ones_like(y_logit_real)))
  d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_fake, 
                                                                      labels = tf.zeros_like(y_logit_fake)))

  d_loss = d_loss_real + d_loss_fake
    
  # optimizer
  d_solver = tf.compat.v1.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
        
  # -------------------- #
  # ----- Training ----- #
  # -------------------- #    
  # Start session and initialize
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())
    
  # Train/test division for both original and generated data
  train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
  train_test_divide(ori_data, generated_data, ori_time, generated_time)
  
  loss = [] 

  # Training step
  for itt in range(iterations):
          
    # Batch setting
    X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
    X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
          
    # Train discriminator
    _, step_d_loss = sess.run([d_solver, d_loss], 
                              feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb}) 
    
    #loss.append(step_d_loss)
               
    
  ## Test the performance on the testing set    
  y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake], 
                                                feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})
    
  y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
  y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)
    
  # Compute the accuracy
  acc = accuracy_score(y_label_final, (y_pred_final>0.5))
  discriminative_score = np.abs(0.5-acc)
    
  return discriminative_score


def compute_predictive_score (ori_data, generated_data):
  """Report the performance of Post-hoc RNN one-step ahead prediction.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    
  Returns:
    - predictive_score: MAE of the predictions on the original data
  """

  # -------------------------- #
  # ----- Initialization ----- #
  # -------------------------- #
  tf.compat.v1.reset_default_graph()
  tf.compat.v1.disable_eager_execution()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Set maximum sequence length and each sequence length (change in case of variable length sequences)
  ori_time = np.full(shape=no, fill_value=seq_len, dtype=np.int)
  ori_max_seq_len = seq_len
  generated_time = np.full(shape=no, fill_value=seq_len, dtype=np.int)
  generated_max_seq_len = seq_len
  max_seq_len = seq_len 
     
  # --------------------------------------------- #
  # ----- RNN Predictive network definition ----- #
  # --------------------------------------------- #

  # Network parameters
  hidden_dim = int(dim/2)
  iterations = 5000
  batch_size = 128

  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x")
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")    
  Y = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")
    
  # Predictor function
  def predictor (x, t):
    """Predictor network.
    
    Args:
      - x: time-series data
      - t: time information
      
    Returns:
      - y_hat: prediction
      - p_vars: predictor variables
    """
    with tf.compat.v1.variable_scope("predictor", reuse = tf.compat.v1.AUTO_REUSE) as vs:
      p_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
      p_outputs, p_last_states = tf.compat.v1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length = t)
      y_hat_logit = tf.compat.v1.layers.dense(p_outputs, 1, activation=None) 
      y_hat = tf.nn.sigmoid(y_hat_logit)
      p_vars = [v for v in tf.compat.v1.all_variables() if v.name.startswith(vs.name)]
    
    return y_hat, p_vars
    
  y_pred, p_vars = predictor(X, T)

  # Loss for the predictor
  p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)

  # optimizer
  p_solver = tf.compat.v1.train.AdamOptimizer().minimize(p_loss, var_list = p_vars)
        
  # -------------------- #
  # ----- Training ----- #
  # -------------------- #  
  # Session start
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())
    
  # Training using Synthetic dataset
  for itt in range(iterations):
          
    # Set mini-batch
    idx = np.random.permutation(len(generated_data))
    train_idx = idx[:batch_size]     
            
    X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
    T_mb = list(generated_time[i]-1 for i in train_idx)
    Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)        
          
    # Train predictor
    _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})        
    
  ## Test the trained model on the original data
  idx = np.random.permutation(len(ori_data))
  train_idx = idx[:no]
    
  X_mb = list(ori_data[i][:-1,:(dim-1)] for i in train_idx)
  T_mb = list(ori_time[i]-1 for i in train_idx)
  Y_mb = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx)
    
  # Prediction
  pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})
    
  # Compute the performance in terms of MAE
  MAE_temp = 0
  for i in range(no):
    MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
    
  predictive_score = MAE_temp / no
    
  return predictive_score


def visualize_PCA(ori_data, generated_data, n_components = 2):
  """Visualize the PCA plot of original and generated data.

  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - n_components: number of principal components to visualize (2D or 3D plot)  
  """
  
  if (n_components != 2 and n_components != 3):
    print('Invalid component number. Please choose either 2 or 3.')
    return

  no, seq_len, dim = np.asarray(ori_data).shape
  sample_size = 1000
  
  idx = np.random.permutation(len(ori_data))[:sample_size]

  real_sample = np.asarray(ori_data)[idx]
  synthetic_sample = np.asarray(generated_data)[idx]

  #for the purpose of comparision we need the data to be n_component-Dimensional. 
  synth_data_reduced = real_sample.reshape(-1, 35)
  stock_data_reduced = np.asarray(synthetic_sample).reshape(-1,35)

  pca = PCA(n_components=n_components)  
  pca.fit(stock_data_reduced)

  pca_real = pd.DataFrame(pca.transform(stock_data_reduced))
  pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

  #Plot the data
  plt.figure(figsize=(10,10)) 
  
  plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:,1].values,
              c='black', alpha=0.2, label='Original')
  plt.scatter(pca_synth.iloc[:,0], pca_synth.iloc[:,1],
              c='red', alpha=0.2, label='Synthetic')
  plt.title('PCA Results')
  plt.legend()


def visualize_TSNE(ori_data, generated_data, n_components = 2):

  """Visualize the t-SNE plot of original and generated data.

  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - n_components: number of components to visualize (2D or 3D plot)  
  """

  if (n_components != 2 and n_components != 3):      
    print('Invalid component number. Please choose either 2 or 3.')
    return

  no, seq_len, dim = np.asarray(ori_data).shape
  sample_size = 1000
  
  idx = np.random.permutation(len(ori_data))[:sample_size]

  real_sample = np.asarray(ori_data)[idx]
  synthetic_sample = np.asarray(generated_data)[idx]  
  
  synth_data_reduced = real_sample.reshape(-1, 35)
  stock_data_reduced = np.asarray(synthetic_sample).reshape(-1,35)

  tsne = TSNE(n_components=n_components, n_iter=300)

  data_reduced = np.concatenate((stock_data_reduced, synth_data_reduced), axis=0)
  tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

  # Plot the data
  plt.figure(figsize=(10,10)) 
  
  plt.scatter(tsne_results.iloc[sample_size:,0], tsne_results.iloc[sample_size:,1],
              c='salmon', alpha=0.2, label='Synthetic')
  plt.scatter(tsne_results.iloc[:sample_size, 0].values, tsne_results.iloc[:sample_size,1].values,
              c='black', alpha=0.2, label='Original')  
  plt.title('t-SNE Results')
  plt.legend()

def compute_MMD(ori_data, generated_data, sample_size = 100):

  # np.random.seed(1)

  idx = np.random.permutation(len(ori_data))[:sample_size]
  real_sample = np.asarray(ori_data)[idx]
  synthetic_sample = np.asarray(generated_data)[idx]

  real_sample = torch.tensor(real_sample)
  synthetic_sample = torch.tensor(synthetic_sample)

  real_sample = real_sample.view(real_sample.shape[0], real_sample.shape[1] * real_sample.shape[2])
  synthetic_sample = synthetic_sample.view(synthetic_sample.shape[0], synthetic_sample.shape[1] * synthetic_sample.shape[2])

  xx, yy, zz = torch.mm(real_sample,real_sample.t()), torch.mm(synthetic_sample,synthetic_sample.t()), torch.mm(real_sample,synthetic_sample.t())

  rx = (xx.diag().unsqueeze(0).expand_as(xx))
  ry = (yy.diag().unsqueeze(0).expand_as(yy))

  K = torch.exp(- 0.2 * (rx.t() + rx - 2*xx))
  L = torch.exp(- 0.2 * (ry.t() + ry - 2*yy))
  P = torch.exp(- 0.2 * (rx.t() + ry - 2*zz))

  beta = (1./(sample_size*(sample_size-1)))
  gamma = (2./(sample_size*sample_size)) 

  return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)
