"""
Train the parameters of the sinogram interpolation network
"""
#pylint: disable=C0103, C0411, C0301, E1101
from utils import int_U_net, list_files, save_image
from sino_utils import get_op
import os
import numpy as np
import tensorflow as tf

ANGLES = 48
PROJS = 800

checkpoint_dir = '/home/hzyuan/CT/checkpoint/interp_%d/'%ANGLES
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
inp_set = list_files('/home/hzyuan/CT/AAPM_dataset/sinos/', name='%d_*.npy'%ANGLES)
out_set = list_files('/home/hzyuan/CT/AAPM_dataset/sinos/', name='%d_*.npy'%(ANGLES * 4))


eval_set = list_files('/home/hzyuan/CT/AAPM_dataset/sinos_test', name = '%d_L*.npy'%ANGLES)
eval_outset = list_files('/home/hzyuan/CT/AAPM_dataset/sinos_test', name = '%d_L*.npy'%(ANGLES * 4))

inp_set = sorted(inp_set, key=lambda x: x.split('/')[-1].split('.')[0])
out_set = sorted(out_set, key=lambda x: x.split('/')[-1].split('.')[0])

print(inp_set[2299])
print(out_set[2299])
inp_dict = {k:i for i, k in enumerate(inp_set)}
out_dict = {k:i for i, k in enumerate(out_set)}

o1,p1 = get_op(48, 800)
o2, p2 = get_op(192, 800)

print(len(inp_dict))

class sint(object):
    def __init__(self, sess, batch_size=8):
        self.sess = sess
        self.batch_size = batch_size
        self.in_h = ANGLES
        self.in_w = PROJS
        self.out_h = ANGLES * 4
        self.out_w = PROJS
        self.build_model()
    def build_model(self):
        """
        building tensorflow graph
        """
        self.inp = tf.placeholder(tf.float32, [None, self.in_h, self.in_w, 1], 'inp')
        self.out = tf.placeholder(tf.float32, [None, self.out_h, self.out_w, 1], 'out')
        self.pred_out = int_U_net(self.inp)
        self.pred_test = int_U_net(self.inp, reuse=True, train=False)
        self.loss = tf.nn.l2_loss(self.out - self.pred_out)
        self.test_loss = tf.nn.l2_loss(self.out - self.pred_test)
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.005
        lr = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 0.96, staircase=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(init)

    def train(self, inp_set=inp_set, out_set=out_set, epoch=10):
        #ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print('training')
        batch_idx = int(len(inp_set)//self.batch_size)
        order = list(range(len(inp_set)))
        for ep in range(epoch):
            np.random.shuffle(order)
            order_dict = {k:m for m, k in enumerate(order)}
            inp_set = sorted(inp_set, key=lambda x: order_dict[inp_dict[x]])
            out_set = sorted(out_set, key=lambda x: order_dict[out_dict[x]])
            for j in range(batch_idx):
                batch_i = inp_set[j*self.batch_size:(j+1)*self.batch_size]
                batch_o = out_set[j*self.batch_size:(j+1)*self.batch_size]
                tmp_i = np.reshape(np.array([np.load(p) for p in batch_i]), [self.batch_size, self.in_h, self.in_w, 1])
                tmp_o = np.reshape(np.asarray([np.load(f) for f in batch_o]), [self.batch_size, self.out_h, self.out_w, 1])

                _, err = self.sess.run([self.opt, self.pred_out], feed_dict={self.inp:tmp_i, self.out:tmp_o})
                print('%d_%d_%f'%(ep, j, err))
                if j%100 == 0:
                    if not j == 0:
                        self.saver.save(self.sess, checkpoint_dir + 'model.ckpt', global_step=ep*int(len(inp_set)/self.batch_size)+j)
    def eval(self, inp_set=eval_set, out_set=eval_outset, epoch=1):
        print('evaluating')
        batch_idx = int(len(inp_set)//self.batch_size)
        for ep in range(epoch):
            for j in range(batch_idx):
                batch_i = inp_set[j*self.batch_size:(j+1)*self.batch_size]
                batch_o = out_set[j*self.batch_size:(j+1)*self.batch_size]
                tmp_i = np.reshape(np.array([np.load(p) for p in batch_i]), [self.batch_size, self.in_h, self.in_w, 1])
                tmp_o = np.reshape(np.array([np.load(f) for f in batch_o]), [self.batch_size, self.out_h, self.out_w, 1])

                err, pred_img = self.sess.run([self.test_loss, self.pred_test], feed_dict={self.inp:tmp_i, self.out:tmp_o})
                pred_sino = pred_img.reshape([self.out_h, self.out_w])

                inp_sino = tmp_i.reshape([self.in_h, self.in_w])
                inp_img = p1(inp_sino)
                pred_img = p2(pred_sino).asarray().reshape([512, 512]).astype(np.float32)

                print('%d_%d_%f'%(ep, j, err))
                save_image('eval_%d.png'%j, pred_img)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
inpaint = sint(sess)
inpaint.train(epoch=100)
