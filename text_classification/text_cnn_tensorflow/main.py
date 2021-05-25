# -- coding: utf-8 -*-

import argparse
import atexit
import logging
from hbconfig import Config
import tensorflow as tf
import sys

sys.path.append("..")
from tensorflow.python import debug as tf_debug
from text_cnn_tensorflow import data_loader
from text_cnn_tensorflow import hook
from text_cnn_tensorflow.model import Model
from text_cnn_tensorflow import utils


def experiment_fn(run_config, params):
    model = Model()  # todo 核心模型
    estimator = tf.estimator.Estimator(  # todo estimator高级api
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        params=params,
        config=run_config)

    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)

    # todo 这些操作完全可以使用tf.data高级api来完成
    train_data, test_data = data_loader.make_train_and_test_set()

    # todo 创建hook钩子，好好学习
    train_input_fn, train_input_hook = data_loader.make_batch(train_data, batch_size=Config.model.batch_size,
                                                              scope="train")
    test_input_fn, test_input_hook = data_loader.make_batch(test_data, batch_size=Config.model.batch_size, scope="test")

    train_hooks = [train_input_hook]
    if Config.train.print_verbose:
        train_hooks.append(hook.print_variables(
            variables=['train/input_0'],
            rev_vocab=get_rev_vocab(vocab),
            every_n_iter=Config.train.check_hook_n_iter))

        train_hooks.append(hook.print_target(
            variables=['train/target_0', 'train/pred_0'],
            every_n_iter=Config.train.check_hook_n_iter))

    if Config.train.debug:
        train_hooks.append(tf_debug.LocalCLIDebugHook())

    eval_hooks = [test_input_hook]
    if Config.train.debug:
        eval_hooks.append(tf_debug.LocalCLIDebugHook())

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=Config.train.train_steps,
        min_eval_frequency=Config.train.min_eval_frequency,
        train_monitors=train_hooks,
        eval_hooks=eval_hooks
    )
    return experiment


def get_rev_vocab(vocab):
    if vocab is None:
        return None
    return {idx: key for key, idx in vocab.items()}


def main(mode):
    params = tf.contrib.training.HParams(**Config.model.to_dict())

    #  todo Replace tf.contrib.learn with tf.estimator.
    run_config = tf.contrib.learn.RunConfig(
        model_dir=Config.train.model_dir,
        save_checkpoints_steps=Config.train.save_checkpoints_steps)

    tf.contrib.learn.learn_runner.run(  # todo 改成estimator高级api 或者 使用TFLearn高级api
        experiment_fn=experiment_fn,  # todo 传入的模型与model_fn一致
        run_config=run_config,  # todo 默认的设置runConfig类
        schedule=mode,  # todo 模式train、eval、predict
        hparams=params  # todo 自己需要带的参数，与param一致
    )


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
    atexit.register(utils.send_message_to_slack, config_name=args.config)

    main(args.mode)
