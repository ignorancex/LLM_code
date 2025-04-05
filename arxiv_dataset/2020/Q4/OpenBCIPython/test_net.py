import argparse

import tensorflow as tf
import time


from loader import DataLoader
from neuralnet.net.cnn.model1.convolutional_network import CNNModel1
from neuralnet.net.cnn.model2.inception_resnet_v2 import CNNModel2


def runTest(loader, cnn_model):
    with tf.name_scope('Model'):
        model_predicted_output, _ = cnn_model.conv_net(cnn_model.x_input)

    # Declare Loss Function (softmax cross entropy)
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=
                                                                      model_predicted_output,
                                                                      labels=cnn_model.y_target))
    # Define loss and optimizer
    with tf.name_scope('AdamOptimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=cnn_model.learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(model_predicted_output, 1), tf.argmax(cnn_model.y_target, 1))

    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    # loader.create_one_big_file("ogg")
    # Launch the graph
    with tf.Session() as sess:
        image, label = loader.inputs()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(cnn_model.logs_path, graph=tf.get_default_graph())

        ckpt = tf.train.get_checkpoint_state(cnn_model.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("There is a no model which has been saved previously in this directory: %s" % cnn_model.model_path)

        step = 1
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # Keep training until reach max iterations
        try:
            step = 0
            start_time = time.time()
            while not coord.should_stop():
                # Run training steps or whatever
                batch_x, batch_y = sess.run([image, label])
                batch_x = batch_x[0:1][0]
                batch_y = batch_y[0:1][0]
                # Run optimization op (backprop)
                _, summary = sess.run([optimizer, merged_summary_op],
                                      feed_dict={cnn_model.x_input: batch_x, cnn_model.y_target: batch_y,
                                                 cnn_model.keep_prob: cnn_model.dropout})
                summary_writer.add_summary(summary, step * loader.batch_size + step)
                loss, acc = sess.run([cost, accuracy], feed_dict={cnn_model.x_input: batch_x,
                                                                  cnn_model.y_target: batch_y,
                                                                  cnn_model.keep_prob: 1.})
                print("Iter " + str(step * loader.batch_size) + ", Minibatch Loss= {:.6f}".format(
                    loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                # TODO save the model as you require...
                saver.save(sess, cnn_model.model_path, global_step=step)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (loader.num_epochs, loader.batch_size))
        finally:
            coord.request_stop()
        coord.join(threads)
        save_path = saver.save(sess, cnn_model.model_path)
        print("Model saved in file: %s" % save_path)
        sess.close()

        print("Optimization Finished!")
        print("Run the command line:\n" \
              "--> tensorboard --logdir=%s " \
              "\nThen open http://0.0.0.0:6006/ into your web browser" % cnn_model.logs_path)



project_dir = "/home/runge/openbci/git/OpenBCI_Python"
dataset_dir = "/home/runge/openbci/git/OpenBCI_Python/build/dataset"
loader = DataLoader(project_dir, dataset_dir)
cnn_model = CNNModel2(project_dir, loader.get_train_config())

runTest(loader, cnn_model)
