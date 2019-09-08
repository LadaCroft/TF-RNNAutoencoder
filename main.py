import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


from models import *
from optimize import Optimize
from data_processing import *
import hyperparameters as hp


########################## Data import + formatting #############################
data, column = import_data(hp.options['input_data'])
data[data.columns[0]] = range(0, len(data))
valid, train, test = train_test_valid_split(
    data, hp.options['train_ratio'])


valid_scaled, train_scaled, test_scaled = valid, train, test

deviation, mean, valid_scaled['scaled'], train_scaled['scaled'], test_scaled['scaled'], train_scaled_mean = zscore(
    valid_scaled[column], train_scaled[column], test_scaled[column])
valid_scaled['scaled'], train_scaled['scaled'], test_scaled['scaled'], scaler = normalization(
    valid_scaled, train_scaled, test_scaled, 'scaled')

original = column
column = 'scaled'
columns = []
columns.append(column)


def first_model(options, train_scaled, test_scaled, valid_scaled, column, columns, original, path):
    num_units = []
    for layer in range(options['num_layers']):
        num_units.append(options['num_hidden'])

    Y_train, X_batch_train, Y_batch_train = batch_whole_vt(
        np.array(train_scaled[columns]), options['steps_enc'], options['steps_dec'])
    Y_valid, X_batch_valid, Y_batch_valid = batch_whole_vt(
        np.array(valid_scaled[columns]), options['steps_enc'], options['steps_dec'])
    Y_test, X_batch_test, Y_batch_test = batch_whole_test(
        np.array(test_scaled[column]), options['steps_enc'], options['steps_dec'])

    X_batch_valid, X_batch_test, X_batch_train, columns = side_channel(
        valid_scaled, train_scaled, test_scaled, Y_batch_valid, Y_batch_train, Y_batch_test, X_batch_valid, X_batch_train, X_batch_test, column, columns, options)

    num_inputs = len(columns)  # 1 s index columns
    num_outputs = 1

    with tf.Graph().as_default():
        ################### Model #####################
        with tf.name_scope('input'):
            encoder_inputs = tf.placeholder(tf.float32, shape=(
                None, options['steps_enc'], num_inputs), name='encoder_inputs')
            # this changes to tensor of steps_enc array - like (step_num,batch_num,elem_num) but these are separate arrays (to be able to iterate over Tensors)
            encoder_inputs_ta = [tf.squeeze(t, [1]) for t in tf.split(
                encoder_inputs, options['steps_enc'], 1)]

            current_batch_size = tf.shape(encoder_inputs_ta)[1]

            decoder_targets = tf.placeholder(
                tf.float32, shape=(None, options['steps_dec'], num_outputs), name='decoder_targets')
            decoder_targets_ta = [tf.squeeze(t, [1]) for t in tf.split(
                decoder_targets, options['steps_dec'], 1)]

        model = init_model(options['model_type'], current_batch_size,
                           encoder_inputs_ta, decoder_targets_ta, num_units, options['cell_type'])

        decoder_outputs = model.prediction
        targets = model.target

        #########################     Objective function     ##############################
        with tf.name_scope('loss'):
            # Objective function = MSE
            loss = tf.reduce_mean(
                tf.square(decoder_outputs - targets), name='loss')
            # for TensorBoard
            loss_train = tf.compat.v1.summary.scalar('loss_train', loss)
            loss_test = tf.compat.v1.summary.scalar('loss_test', loss)
            loss_valid = tf.compat.v1.summary.scalar('loss_valid', loss)
            loss_batch_train = tf.compat.v1.summary.scalar(
                'loss_batch_train', loss)

        #############################      Training      ##################################
        training = Optimize(loss, gradient_clipping=False)

        #############################   TF preparation   ##################################
        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.compat.v1.summary.merge_all()

        # TB from folder "done": tensorboard --logdir=run1:./tmp/seq2seq/ --port 6006
        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.Saver(save_relative_paths=True)
            # variables need to be initialized before we can use them
            sess.run(tf.compat.v1.global_variables_initializer())

            # create log writer object
            writer = tf.compat.v1.summary.FileWriter(
                path, graph=tf.compat.v1.get_default_graph())
            save_hyperparameters(options, path)

            # https://github.com/mmuratarat/handson-ml/blob/master/11_deep_learning.ipynb
            max_checks_without_progress = options['max_patience']
            checks_without_progress = 0
            best_loss = np.infty

            try:
                # perform training cycles
                for iteration in range(0, options['max_epochs']+1):

                    x_batch, y_batch = next_batch(
                        np.array(valid_scaled[column]), np.array(train_scaled[column]), options)

                    # Scalars for TensorBoard
                    mse_train, summary_train = sess.run([loss, loss_train], feed_dict={
                        encoder_inputs: X_batch_train, decoder_targets: Y_batch_train})
                    writer.add_summary(summary_train, iteration)
                    mse_test, summary_test = sess.run([loss, loss_test], feed_dict={
                        encoder_inputs: X_batch_test, decoder_targets: Y_batch_test})
                    writer.add_summary(summary_test, iteration)
                    mse_valid, summary_valid = sess.run([loss, loss_valid], feed_dict={
                        encoder_inputs: X_batch_valid, decoder_targets: Y_batch_valid})
                    writer.add_summary(summary_valid, iteration)
                    if iteration == 0:
                        mse_batch_train, summary_batch_train = sess.run([loss, loss_batch_train], feed_dict={
                            encoder_inputs: x_batch, decoder_targets: y_batch})
                        writer.add_summary(summary_batch_train, iteration)
                    print(iteration, '.iteration - MSE_train:',
                          mse_train, ', MSE_test', mse_test)

                    if iteration % 100 == 0:
                        # Compute the predictions
                        train_prediction = sess.run(model.prediction, feed_dict={
                                                    encoder_inputs: X_batch_train})
                        test_prediction = sess.run(model.prediction, feed_dict={
                            encoder_inputs: X_batch_test})
                        Y_train_pred, Y_test_pred = train_test_form(
                            train_prediction, test_prediction, Y_train, Y_test)

                        # Visualize
                        subplot(Y_train, Y_train_pred, Y_test,
                                Y_test_pred, mse_train, mse_test, path, iteration)
                        # Save
                        save_data(path, 'train_pred', Y_train_pred)
                        save_data(path, 'test_pred', Y_test_pred)

                    model_path = path + '/my_model'
                    if mse_valid < best_loss:
                        best_loss = mse_valid
                        checks_without_progress = 0
                        saver.save(sess, model_path)
                    else:
                        checks_without_progress += 1
                        if checks_without_progress > max_checks_without_progress:
                            print('Early stopping!')
                            saver.restore(sess, model_path)
                            iteration = str(
                                iteration - 1 - max_checks_without_progress)
                            print('Model restored.')
                            break

                    _, summary_batch_train, _ = sess.run([loss, loss_batch_train, training.training_op], feed_dict={
                        encoder_inputs: x_batch, decoder_targets: y_batch})
                    writer.add_summary(summary_batch_train, iteration)
                print('done')

            except KeyboardInterrupt:
                print('training interrupted')

            train_prediction = sess.run(model.prediction, feed_dict={
                encoder_inputs: X_batch_train})
            test_prediction = sess.run(model.prediction, feed_dict={
                encoder_inputs: X_batch_test})
            # rescale
            Y_train_pred, Y_test_pred = train_test_form(
                train_prediction, test_prediction, Y_train, Y_test)
            Y_train_pred = inverse_normalization(
                scaler, Y_train_pred.reshape(-1, 1))
            Y_test_pred = inverse_normalization(
                scaler, Y_test_pred.reshape(-1, 1))
            Y_train_pred = inverse_zscore(Y_train_pred, mean, deviation)
            Y_test_pred = inverse_zscore(Y_test_pred, mean, deviation)

            # to have back the original data before scaling and normalizing
            if options['steps_enc'] == options['steps_dec']:
                Y_train = np.array(
                    train[original].iloc[options['steps_enc']:]).reshape(-1, 1)
                Y_test = np.array(
                    test[original].iloc[options['steps_enc']:]).reshape(-1, 1)
            else:
                Y_train = np.array(
                    train[original].iloc[-len(Y_train_pred):]).reshape(-1, 1)
                Y_test = np.array(
                    test[original].iloc[options['steps_enc']:options['steps_enc'] + len(Y_test_pred)]).reshape(-1, 1)

            mean_test = np.mean(Y_test)
            std_test = np.std(Y_test, ddof=1)
            print('data:', options['input_data'], 'mean:',
                  mean_test, 'standard deviation', std_test, 'n', len(Y_test))

            mse_train = np.mean((Y_train-Y_train_pred)**2)
            rmse_train = np.sqrt(mse_train)
            mse_test = np.mean((Y_test-Y_test_pred)**2)
            rmse_test = np.sqrt(mse_test)
            save_data(path, 'finalTrain', Y_train_pred)
            save_data(path, 'finalTest', Y_test_pred)

            final_subplot(Y_train, Y_train_pred, Y_test, Y_test_pred,
                          mse_train, mse_test, rmse_train, rmse_test, path, iteration)
            rmse_train_parts = plot_parts(
                options['steps_dec'], Y_train, Y_train_pred, 'Train', path)
            rmse_test_parts = plot_parts(
                options['steps_dec'], Y_test, Y_test_pred, 'Test', path)

    return model_path


path = make_path(hp.options)
print('\n \n \n \n MODEL: ', path)

model_path = first_model(hp.options, train_scaled, test_scaled,
                         valid_scaled, column, columns, original, path)
print('\n model_path:', model_path)
