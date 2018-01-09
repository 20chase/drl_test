import numpy as np
import tensorflow as tf
import tensorlayer as tl
import utils as U


class Master(object):
    def __init__(self, sess, ob_ph, options, name):
        self.sess = sess
        self.ob_ph = ob_ph
        self.options = options
        
        self.pd, self.vpred, self.a_net, self.c_net = self._build_net(name)

    def _build_net(self, name):
        with tf.variable_scope(name)
            self.scope = tf.get_variable_scope().name
            a_net = tl.layers.InputLayer(self.ob_ph, name='a_input')
            a_net = tl.layers.DenseLayer(a_net, n_units=64, act=tf.nn.tanh,
                name='a_h1')
            a_net = tl.layers.DenseLayer(a_net, n_units=64, act=tf.nn.tanh,
                name='a_h2')
            a_net = tl.layers.DenseLayer(a_net, n_units=self.options, 
                name='a_output')

            c_net = tl.layers.InputLayer(self.ob_ph, name='c_input')
            c_net = tl.layers.DenseLayer(c_net, n_units=64, act=tf.nn.tanh,
                name='c_h1')
            c_net = tl.layers.DenseLayer(c_net, n_units=64, act=tf.nn.tanh,
                name='c_h2')
            c_net = tl.layers.DenseLayer(c_net, n_units=1, name='c_output')

            pdtype = U.CategoricalPdType(self.options)
            pd = pdtype.pdfromflat(a_net.outputs)
            vpred = c_net.outputs[:, 0]

        return pd, vpred, a_net, c_net

    def step(self, ob):
        feed_dict = {self.ob_ph: ob}
        return self.sess.run([self.pd.sample(), self.vpred], feed_dict=feed_dict)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class SubPolicy(object):
    def __init__(self, sess, ob_ph, ac_dim, idx, name):
        self.sess = sess
        self.ob_ph = ob_ph
        self.ac_dim = ac_dim

        self.mean, self.vpred, self.pd, self.a_net, self.c_net = self._build_net(idx, name)

    def _build_net(self, i, name):
        with tf.variable_scope('{}_{}'.format(name, i)):
            a_net = tl.layers.InputLayer(self.ob_ph, name='a{}_input'.format(i))
            a_net = tl.layers.DenseLayer(a_net, n_units=64, act=tf.nn.tanh,
                name='a{}_h1'.format(i))
            a_net = tl.layers.DenseLayer(a_net, n_units=64, act=tf.nn.tanh,
                name='a{}_h2'.format(i))
            a_net = tl.layers.DenseLayer(a_net, n_units=self.ac_dim, act=tf.nn.tanh,
                name='a{}_output'.format(i))

            c_net = tl.layers.InputLayer(self.ob_ph, name='c{}_input'.format(i))
            c_net = tl.layers.DenseLayer(c_net, n_units=64, act=tf.nn.tanh,
                name='c{}_h1'.format(i))
            c_net = tl.layers.DenseLayer(c_net, n_units=64, act=tf.nn.tanh,
                name='c{}_h2'.format(i))
            c_net = tl.layers.DenseLayer(c_net, n_units=1, name='c{}_output'.format(i))

            logstd = tf.get_variable(name='a{}_logstd'.format(i), shape=[1, self.ac_dim],
                initializer=tf.zeros_initializer())

        mean = a_net.outputs
        vpred = c_net.outputs

        pdparam = tf.concat([mean, mean*0.0 + logstd], axis=1)

        pdtype = U.DiagGaussianPdType(self.ac_dim)
        pd = pdtype.pdfromflat(pdparam)

        return mean, vpred[:, 0], pd, a_net, c_net

    def step(self, ob):
        feed_dict = {self.ob_ph: ob}
        return self.sess.run([self.pd.sample(), self.vpred], feed_dict=feed_dict)


class Agent(object):
    def __init__(self, sess, args, ob_shape, ac_shape, options):
        self.sess = sess
        self.args = args
        self.ob_dim = ob_shape.shape[0]
        self.ac_dim = ac_shape.shape[0]
        self.options = options

        self._build_ph()
        self.master = Master(sess, self.ob_ph, options, name='master')
        self.old_master = Master(sess, self.ob_ph, options, name='old_master')
        self.subpolicies = [SubPolicy(sess, self.ob_ph, ac_dim, idx, 'subpolicy') for idx in range(options)]
        self.old_subpolicies = [SubPolicy(sess, self.ob_ph, ac_dim, idx, 'old_subpolicy') for idx in range(options)]

    def _build_ph(self):
        self.ob_ph = tf.placeholder(tf.float32, [None, self.ob_dim], 'ob_ph')
        self.op_ph = tf.placeholder(tf.int32, [None], 'op_ph')
        self.ac_ph = tf.placeholder(tf.float32, [None, self.ac_dim], 'ac_ph')
        self.adv_ph = tf.placeholder(tf.float32, [None, ], 'adv_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')

        self.lr_ph = tf.placeholder(tf.float32, name='lr_ph')
        self.sub_lr_ph = tf.placeholder(tf.float32, name='sub_lr_ph')

    def _build_loss(self, pi, oldpi, ac, adv, ret, clip_param):
        ratio = tf.exp(pi.pd.neglogp(ac) - oldpi.pd.neglogp(ac))
        surr1 = ratio * adv
        surr2 = tf.clip_by_value(ratio, 1. - clip_param, 1. + clip_param) * adv
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
        vfloss1 = tf.square(pi.vpred - ret)
        vpredclipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -clip_param, clip_param)
        vfloss2 = tf.square(vpredclipped - ret)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vfloss1, vfloss2))
        total_loss = pol_surr + vf_loss
        return total_loss

    def _build_master_training(self):
        with tf.variable_scope('master_loss'):
            total_loss = self._build_loss(self.master, self.old_master, self.op_ph, self.adv_ph, self.ret_ph, 0.2)
        
        params = self.master.get_trainable_variables()
        grads = tf.gradients(total_loss, params)
        if self.args.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.args.max_grad_norm)

        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=self.lr_ph)
        opt = trainer.apply_gradients(grads)
        return opt









