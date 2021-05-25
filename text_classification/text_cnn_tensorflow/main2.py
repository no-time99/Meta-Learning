# -- coding: utf-8 -*-

import argparse
import atexit
import logging
from hbconfig import Config
import tensorflow as tf
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # 指定使用哪些GPU
sys.path.append("..")
from text_cnn_tensorflow import data_loader
from text_cnn_tensorflow import hook
from text_cnn_tensorflow.model import Model

tf.logging.set_verbosity(tf.logging.DEBUG)


def input_fn():
    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)
    train_data, test_data = data_loader.make_train_and_test_set()

    with tf.name_scope("train"):
        X, y = train_data
        # X, y = test_data  # todo 测试集数据输入

        data_set = tf.data.Dataset.from_tensor_slices((X, y))
        data_set = data_set.repeat(None)
        data_set = data_set.shuffle(buffer_size=1000)
        data_set = data_set.batch(128)

    # iterator = data_set.make_initializable_iterator()
    # next_X, next_y = iterator.get_next()

    # tf.identity(next_X[0], 'input_0')
    # tf.identity(next_y[0], 'target_0')

    # Return batched (features, labels)
    # return next_X, next_y
    return data_set


def model_fn(features, labels, mode, params):
    return Model().model_fn(mode, features, labels, params)


def main(_):
    params = tf.contrib.training.HParams(**Config.model.to_dict())

    # 优化GPU
    session_config = tf.ConfigProto(
        log_device_placement=True,
        inter_op_parallelism_threads=0,
        allow_soft_placement=True
    )
    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
        model_dir=Config.train.model_dir,
        save_checkpoints_steps=Config.train.save_checkpoints_steps,
        session_config=session_config
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        # model_dir=Config.train.model_dir,
        params=params,
        config=run_config)

    estimator.train(input_fn=input_fn)
    predictions = estimator.predict(input_fn=input_fn)
    for pre in predictions:
        print(pre)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='kaggle_movie_review', help='config file name')
    parser.add_argument('--mode', type=str, default='train', help='Mode (train/test/train_and_evaluate)')
    args = parser.parse_args()
    print(args.__dict__)
    # tf.logging._logger.setLevel(logging.INFO) # 弃用
    tf.logging.set_verbosity(logging.INFO)

    # Print Config setting
    Config(args.config)
    print("Config: ", Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    # After terminated Notification to Slack
    # atexit.register(utils.send_message_to_slack, config_name=args.config)

    tf.app.run()
