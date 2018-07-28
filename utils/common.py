#########################################################################################
###################################### Data loader ######################################
#########################################################################################

from utils.preprocessing import *
import pandas as pd

def load_data(path_to_data='data/', imgs_path='data/imgs/', 
              path_to_desc='data/train_descriptors_augmented.npz'):
    
    df = pd.read_csv(path_to_data+'pairs_list_augmented.csv')

    train_imgs_source = np.zeros((1000, 5, 112, 112, 3))
    for i, img_names in enumerate(df.source_imgs):
        for j, img_name in enumerate(img_names.split('|')):
            img = Image.open(os.path.join(imgs_path, img_name))
            arr = preprocess_img(img)
            train_imgs_source[i, j] = arr[0].transpose([1, 2, 0])
        
    train_imgs_target = np.zeros((1000, 10, 112, 112, 3))
    for i, img_names in enumerate(df.target_imgs):
        for j, img_name in enumerate(img_names.split('|')):
            img = Image.open(os.path.join(imgs_path, img_name))
            arr = preprocess_img(img)
            train_imgs_target[i, j] = arr[0].transpose([1, 2, 0])
            
    desc_source = np.float32(np.load(path_to_desc)['source'])
    desc_target = np.float32(np.load(path_to_desc)['target'])
    
    return train_imgs_source, train_imgs_target, desc_source, desc_target

#########################################################################################
##################################### GPU management ####################################
#########################################################################################

import os
import tensorflow as tf

def gpu_config(gpu_id):
    if (gpu_id == -1):
        config = tf.ConfigProto()
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads= 1
    return config

#########################################################################################
#################################### Neural Networks ####################################
#########################################################################################

from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Lambda, BatchNormalization, Average
import keras.backend as K

class FaceNetFinetuned:
    
    def __init__(self, path_to_models, model_name, bn=False):
        self.load_model(path_to_models, model_name, bn=bn)
        self.create_graph()
    
    def load_model(self, path_to_models='models/', model_name='bs256.h5', bn=False):

        model = load_model(path_to_models + 'facenet_keras.h5')
        
        model_chop = Model(model.inputs, model.layers[-3].output)
        inp = Input((112, 112, 3))
        inp_resize = Lambda(lambda x: tf.image.resize_images(x, (160, 160)))(inp)
        outp = model_chop(inp_resize)
        outp1 = Dense(512)(outp)
        if bn: outp1 = BatchNormalization()(outp1)
        normalized = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(outp1)
        self.model = Model(inputs=inp, outputs=normalized)
        self.model.load_weights(path_to_models + model_name)
        
    def create_graph(self):
        self.input_images = tf.placeholder(tf.float32, (1, 112, 112, 3))
        self.target_desc = tf.placeholder(tf.float32, (None, 512))
        self.model_desc = self.model(self.input_images)
        
        self.loss = tf.reduce_sum(tf.square(self.target_desc-self.model_desc), axis=1)
        self.loss = tf.reduce_sum(tf.sqrt(self.loss + 1e-8))

        grad = tf.gradients(self.loss, self.input_images)[0]
        self.grad = grad / tf.reduce_sum(tf.abs(grad))
        
    def get_grad(self, sess, input_image, target_desc):
        feed_dict = {self.input_images:input_image,
                     self.target_desc:target_desc,
                     K.learning_phase():0}
        grad = [np.round(sess.run(self.grad, feed_dict)[0], 5)]
        return grad
    
class ModelEnsemble:
    
    def __init__(self, neural_nets):
        self.make_model(neural_nets)
        self.create_graph()
        
    def make_model(self, neural_nets):
        model_input = Input((112, 112, 3))
        outs = [nn.model(model_input) for nn in neural_nets]
        ave = Average()(outs)
        ave = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(ave)
        self.ens = Model(inputs=model_input, outputs=ave)
        
    def create_graph(self):
        self.input_images = tf.placeholder(tf.float32, (1, 112, 112, 3))
        self.target_desc = tf.placeholder(tf.float32, (None, 512))
        self.model_desc = self.ens(self.input_images)
        
        self.loss = tf.reduce_sum(tf.square(self.target_desc-self.model_desc), axis=1)
        self.loss = tf.reduce_sum(tf.sqrt(self.loss + 1e-8))
        
        self.grads = tf.gradients(self.loss, self.input_images)[0]
        
    def get_grads(self, sess, input_image, target_desc):
        feed_dict = {self.input_images:input_image,
                     self.target_desc:target_desc,
                     K.learning_phase():0}
        grads = sess.run(self.grads, feed_dict)
        return grads
    
#########################################################################################
######################################## Attacker #######################################
#########################################################################################

class Attacker:
    
    def __init__(self, black_box, neural_net, tf_session):
        self.bb = black_box
        self.nn = neural_net
        self.sess = tf_session
    
    def get_ssim(self, img, eps):
        img_attacked = img + eps
        img1 = np.clip(denormalize(img_attacked.copy()), 0, 1)
        img2 = np.clip(denormalize(img.copy()), 0, 1)
        return calc_ssim(img1, img2)
    
    def ssim_project(self, img, eps, tol):
        coeff = 1.0
        while self.get_ssim(img, eps*coeff) < 0.95 + tol:
            coeff = coeff - 1e-3
        return img + coeff * eps
    
    def get_grads(self, img, targets):
        grads = np.squeeze(self.nn.get_grads(self.sess, img[None], targets))
        return grads
    
    def get_loss(self, img, targets):
        desc = self.bb.submit(tf2bb(img).copy()).squeeze()
        loss = np.linalg.norm(targets - desc, axis=1).mean()
        return loss
    
    def nesterov_attack(self, img, targets, maxiter=10, gamma=0.9, 
                        alpha0=0.1, decay=0.99, tol=2e-4):
        log, loss_log = [], []
        img_init = img.copy()
        m = np.zeros((112, 112, 3), dtype=np.float32) #momentum
        alpha = alpha0
        
        for i in range(maxiter):
            alpha = alpha * decay
            m = gamma * m + alpha * self.get_grads(img-gamma*m, targets)
            img = img - m
            if (self.get_ssim(img_init, img-img_init) < 0.95 + tol):
                img = self.ssim_project(img_init, img-img_init, tol)
            log.append(img)
            loss_log.append(self.get_loss(img, targets))
        best_idx = np.argmin(np.array(loss_log))
        return log[best_idx], loss_log[best_idx]