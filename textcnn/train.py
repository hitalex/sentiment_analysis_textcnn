#! /usr/bin/env python

#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_input_helper as data_helpers
from text_cnn import TextCNN
import math
from tensorflow.contrib import learn

# Parameters

# ==================================================
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("train_data_file", "/var/proj/sentiment_analysis/data/cutclean_tiny_stopword_corpus10000.txt", "Data source for the positive data.")
#tf.flags.DEFINE_string("train_data_file", "../data/cutclean_label_corpus10000.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("train_data_file", "../datasets/phase1/train.csv", "Train datasets for phase1")
tf.flags.DEFINE_string("train_label_data_file", "", "Data source for the label data.")
tf.flags.DEFINE_string("w2v_file", "../data/vectors.bin", "w2v_file path")
tf.flags.DEFINE_string("chi_word_vector_file", "../word-embeddings-data/sgns.weibo.bigram-char", "embedding file path")

# Model Hyperparameters

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)") # depth of per filter size
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def load_data(embedding_model, dataset_pickle_path):
    """Loads starter word-vectors and train/dev/test data.
    Input:
        embedding_model: WordEmbeddingModel对象，这里并不特定指定使用某个词向量模型
        dataset_pickle_path: 数据集路径，可以是针对某个subject的数据集
    Output:
        x_train, y_train, x_dev, y_dev: 训练集样本以及标签，验证集样本以及标签
        vocab_size: 词典大小
    """
    import pickle
    # Load the starter word vectors
    print("Loading data...")
    #x_text, y = data_helpers.load_data_and_labels(FLAGS.train_data_file)

    # 当前只读入subject类别信息
    #content_id_list, x_text, y, _ = pickle.load(open("../pickles/preprocessed-train-dataset.pickle"))
    content_id_list, x_text, y = pickle.load(open(dataset_pickle_path, 'rb'))

    # 每句话最多的单词数
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print ('len(x) = ',len(x_text),' ',len(y))
    print(' max_document_length = ' , max_document_length)

    x = []
    vocab_size = 0
    if embedding_model is None:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        vocab_size = len(vocab_processor.vocabulary_)

        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(int(time.time()))))
        vocab_processor.save("vocab.txt")
        print( 'save vocab.txt')
    else:
        # embedding_model.vocab_hash 相对于一个dict
        print('Using word embeddings!')
        x = data_helpers.get_text_idx(x_text, embedding_model.vocab_hash, max_document_length)
        vocab_size = len(embedding_model.vocab_hash)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    return x_train,x_dev,y_train,y_dev,vocab_size


def train(embedding_model):
    # Training
    # ==================================================
    print('Loading data...')

    x_train, x_dev, y_train, y_dev ,vocab_size= load_data(embedding_model)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn = TextCNN(
                embedding_model,
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_size,
                #embedding_size = FLAGS.embedding_dim,
                embedding_size = embedding_model.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            # minimize 函数包含 compute_gradients and apply_gradients
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)

            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            # Train Summaries, 聚合所有关于训练的summary
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                # _, step, summaries, loss, accuracy,(w,idx) = sess.run(
                #     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,cnn.get_w2v_W()],
                #     feed_dict)

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # print w[:2],idx[:2]
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }

                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            def dev_test():
                batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                for batch_dev in batches_dev:
                    x_batch_dev, y_batch_dev = zip(*batch_dev)
                    dev_step(x_batch_dev, y_batch_dev, writer=dev_summary_writer)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                # Training loop. For each batch...
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_test()

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    #w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
    #train(w2v_wr.model)

    import sys
    dataset_pickle_path = sys.argv[1]
    
    print('Loading word embedding model...')
    embedding_model = data_helpers.WordEmbeddingModel(FLAGS.chi_word_vector_file)

    train(embedding_model)

