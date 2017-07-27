import numpy as np
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn import metrics

def norm_clipping(params_grad, threshold):
    # calculate norm of gradients
    norm_val = 0.0
    for i in range(len(params_grad[0])):
        for grads in params_grad:
            norm_val += tf.pow(tf.norm(grads[i])[0], 2)
        norm_val = tf.sqrt(norm_val)
    norm_val /= float(len(params_grad[0]))

def train(net, params, q_data, qa_data, label):
    # the number of batch
    N = int(math.floor(len(q_data) / params.batch_size))
    # transpose data matrix
    q_data = q_data.T  # Shape: (seqlen, num_student)
    qa_data = qa_data.T  # Shape: (200,648)
    # shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]
    # list of pred and target
    pred_list = []
    target_list = []

    # progress bar
    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)

    for idx in xrange(N):
        # show progress bar
        if params.show: bar.next()

        # make mini-batch for training
        q_one_seq = q_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

        target = qa_one_seq[:, :]
        target = target.astype(np.int)
        target = (target - 1) / params.n_question
        target = target.astype(np.float)  # correct: 1.0; wrong 0.0; padding -1.0

        # TODO: make placeholder in net.forward and feed inputs and target
        """
        input_q = mx.nd.array(input_q)
        input_qa = mx.nd.array(input_qa)
        target = mx.nd.array(target)
        data_batch = mx.io.DataBatch(data=[input_q, input_qa], label=[target])
        """
        # VRAM limitation for efficient deployment
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        sess.run(tf.global_variables_initializer())
        # define saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        """
        net.forward(data_batch, is_train=True)
        pred = net.get_outputs()[0].asnumpy()  # (seqlen * batch_size, 1)
        net.backward()
        net.update()
        """
        train_feed = data_batch
        _, pred = sess.run([opt, pred], train_feed)
        norm_clipping(net._exec_group.grad_arrays, params.maxgradnorm)

        target = target.asnumpy().reshape((-1,))  # correct: 1.0; wrong 0.0; padding -1.0

        nopadding_index = np.flatnonzero(target != -1.0)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    if params.show: bar.finish()

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    """
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    """

    loss = tf.nn.softmax_cross_entropy_with_logits(all_target, all_pred)
    auc = tf.metrics.auc(all_target, all_pred)
    accuracy = tf.metrics.accuracy(all_target, all_pred)

    return loss, accuracy, auc


def test(net, params, q_data, qa_data, label):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    q_data = q_data.T  # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []
    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)

    count = 0
    element_count = 0
    for idx in xrange(N):
        if params.show: bar.next()

        inds = np.arange(idx * params.batch_size, (idx + 1) * params.batch_size)
        q_one_seq = q_data.take(inds, axis=1, mode='wrap')
        qa_one_seq = qa_data.take(inds, axis=1, mode='wrap')
        # print 'seq_num', seq_num

        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)
        target = qa_one_seq[:, :]
        target = target.astype(np.int)
        target = (target - 1) / params.n_question
        target = target.astype(np.float)  # correct: 1.0; wrong 0.0; padding -1.0
        """
        input_q = mx.nd.array(input_q)
        input_qa = mx.nd.array(input_qa)
        target = mx.nd.array(target)
        
        data_batch = mx.io.DataBatch(data=[input_q, input_qa], label=[])
        net.forward(data_batch, is_train=False)
        pred = net.get_outputs()[0].asnumpy()
        """
        # VRAM limitation for efficient deployment
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        sess.run(tf.global_variables_initializer())
        # define saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

        train_feed = data_batch
        _, pred = sess.run([opt, pred], train_feed)

        target = target.asnumpy()
        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            target = target[:, :real_batch_size]
            pred = pred.reshape((params.seqlen, params.batch_size))[:, :real_batch_size]
            pred = pred.reshape((-1,))
            count += real_batch_size
        else:
            count += params.batch_size

        target = target.reshape((-1,))  # correct: 1.0; wrong 0.0; padding -1.0
        nopadding_index = np.flatnonzero(target != -1.0)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        # print avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    if params.show: bar.finish()
    assert count == seq_num

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    """
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    """

    loss = tf.nn.softmax_cross_entropy_with_logits(all_target, all_pred)
    auc = tf.metrics.auc(all_target, all_pred)
    accuracy = tf.metrics.accuracy(all_target, all_pred)

    return loss, accuracy, auc

# methods for loading and saving checkpoints of the model
def load_checkpoint(sess, saver):
    #ckpt = tf.train.get_checkpoint_state('save')
    #if ckpt and ckpt.model_checkpoint_path:
    #saver.restore(sess, tf.train.latest_checkpoint('save'))
    ckpt = 'savename'+'.ckpt'
    saver.restore(sess, './save/' + ckpt)
    print 'checkpoint {} loaded'.format(ckpt)
    return


def save_checkpoint(sess, saver, g_ep, d_ep):
    checkpoint_path = os.path.join('save', 'savename'+'.ckpt')
    saver.save(sess, checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    return