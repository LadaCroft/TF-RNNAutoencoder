import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from statsmodels.tsa.seasonal import seasonal_decompose
import os.path
import seaborn as sns
sns.set()

import hyperparameters as hp


###########################   Import data   ############################
def import_data(input_data):
    """
    this function imports the chosen dataset
    returns the data and the column name of the 
    predicted variable 
    """
    if input_data == 'ecg':
        data = pd.read_csv('Data\ecg.csv', parse_dates=True)
        data = data.drop(data.index[0])
        data['ECG'] = pd.to_numeric(data['ECG'])
        data.set_index('Elapsed time')
        data.index.freq = '2ms'
        column = 'ECG'
    elif input_data == 'cos_mod':
        data = pd.read_csv('Data\cos_mod.csv', parse_dates=True)
        data['fx'] = pd.to_numeric(data['fx'])
        data.set_index('x')
        column = 'fx'
    elif input_data == 'corn':
        data = pd.read_csv('Data\corn.csv', parse_dates=True)
        input_data = 'corn'
        data.index.freq = 'B'
        data.set_index('date')
        column = 'value'

    return data, column


########################### Formatting data ############################

def train_test_form(train_prediction, test_prediction, Y_train, Y_test):
    """
    this function reshapes the variables for plotting
    """
    Y_train_pred = np.ravel(train_prediction)
    Y_train_pred = Y_train_pred[:len(Y_train)]
    Y_test_pred = np.ravel(test_prediction)
    Y_test_pred = Y_test_pred[:len(Y_test)]

    return Y_train_pred, Y_test_pred


def train_test_valid_split(data, train_ratio):
    """
    this function splits the data into
    validation, training and test dataset
    """
    valid_index = int(len(data)*(train_ratio/(train_ratio*10)))
    train_index = int(len(data)*train_ratio)
    valid = data[:valid_index]
    train = data[valid_index:train_index]
    test = data[train_index:]

    return valid,train, test


def side_channel(valid_scaled, train_scaled, test_scaled, Y_batch_valid, Y_batch_train, Y_batch_test, X_batch_valid, X_batch_train,X_batch_test,column, columns, options):
    """
    this function returns reshaped observations
    if the side-channel method is chosen, the data are concatenated
    with the new variable
    """
    if options['LR'] == True:
        # valid
        border_valid = len(valid_scaled) - len(np.ravel(Y_batch_valid)) - options['steps_enc']
        batches_valid = len(Y_batch_valid)
        slope_valid, intercept_valid = lin_reg_valid(options['LR_part'], valid_scaled['scaled'], border_valid, batches_valid, options['steps_enc'], options['steps_dec'])
        X_batch_valid = np.ravel(X_batch_valid).reshape(-1, 1)
        X_batch_valid = np.concatenate(
            (X_batch_valid, slope_valid[:len(X_batch_valid)],intercept_valid[:len(X_batch_valid)]), axis=1)
        X_batch_valid = X_batch_valid.reshape(-1, options['steps_enc'], 3)

        # train
        border_train = len(train_scaled) - len(np.ravel(Y_batch_train)) - options['steps_enc']
        batches_train = len(Y_batch_train)
        slope_train,intercept_train = lin_reg_train_test(options['LR_part'], valid_scaled['scaled'], train_scaled['scaled'], border_train, batches_train, options['steps_enc'], options['steps_dec'])
        X_batch_train = np.ravel(X_batch_train).reshape(-1, 1)
        X_batch_train = np.concatenate(
            (X_batch_train, slope_train[:len(X_batch_train)],intercept_train[:len(X_batch_train)]), axis=1)
        X_batch_train = X_batch_train.reshape(-1, options['steps_enc'], 3)

        # test
        border_test = len(test_scaled) - len(np.ravel(Y_batch_test)) - options['steps_enc']
        batches_test = len(Y_batch_test)
        slope_test, intercept_test = lin_reg_train_test(options['LR_part'], train_scaled['scaled'], test_scaled['scaled'], border_test, batches_test, options['steps_enc'], options['steps_dec'])
        X_batch_test = np.ravel(X_batch_test).reshape(-1, 1)
        X_batch_test = np.concatenate(
            (X_batch_test, slope_test[:len(X_batch_test)],intercept_test[:len(X_batch_test)]), axis=1)
        X_batch_test = X_batch_test.reshape(-1, options['steps_enc'], 3)

        columns.append('slopes')
        columns.append('intercepts')


    if options['means'] == True:
        # valid
        border_valid = len(valid_scaled) - len(np.ravel(Y_batch_valid)) - options['steps_enc']
        batches_valid = len(Y_batch_valid)
        mean_valid = mean_split_valid(options['mean_part'], valid_scaled['scaled'], border_valid, batches_valid, options['steps_enc'], options['steps_dec'])
        X_batch_valid = np.ravel(X_batch_valid).reshape(-1, 1)
        X_batch_valid = np.concatenate(
            (X_batch_valid, mean_valid[:len(X_batch_valid)]), axis=1)
        X_batch_valid = X_batch_valid.reshape(-1, options['steps_enc'], 2)

        # train
        border_train = len(train_scaled) - len(np.ravel(Y_batch_train)) - options['steps_enc']
        batches_train = len(Y_batch_train)
        mean_train = mean_split_train_test(options['mean_part'], valid_scaled['scaled'], train_scaled['scaled'], border_train, batches_train, options['steps_enc'], options['steps_dec'])
        X_batch_train = np.ravel(X_batch_train).reshape(-1, 1)
        X_batch_train = np.concatenate(
            (X_batch_train, mean_train[:len(X_batch_train)]), axis=1)
        X_batch_train = X_batch_train.reshape(-1, options['steps_enc'], 2)

        # test
        border_test = len(test_scaled) - len(np.ravel(Y_batch_test)) - options['steps_enc']
        batches_test = len(Y_batch_test)
        mean_test = mean_split_train_test(options['mean_part'], train_scaled['scaled'], test_scaled['scaled'], border_test, batches_test, options['steps_enc'], options['steps_dec'])
        X_batch_test = np.ravel(X_batch_test).reshape(-1, 1)
        X_batch_test = np.concatenate(
            (X_batch_test, mean_test[:len(X_batch_test)]), axis=1)
        X_batch_test = X_batch_test.reshape(-1, options['steps_enc'], 2)

        columns.append('mean')

    

    return X_batch_valid,X_batch_test,X_batch_train, columns


def lin_reg_valid(LR_window, valid_scaled, border, batches, steps_enc, steps_dec):
    """
    the function computes the slopes and intercepts for the validation dataset
    """

    LR_deriv = []
    LR_bias = []
    for part in range(batches):
        current_start = border + part*steps_dec
        if (current_start + steps_enc) < LR_window:
            for_LR = valid_scaled[:current_start + steps_enc]

        else:
            for_LR = valid_scaled[current_start -
                                    (LR_window - steps_enc):current_start + steps_enc]

        X = range(len(for_LR))
        X = np.array(X)
        X = X[:, None]

        reg = linear_model.LinearRegression().fit(X, for_LR)
        reg.score(X, for_LR)
        slope = reg.coef_
        intercept = reg.intercept_
        
        part_slope = np.repeat(slope, steps_enc)
        part_intercept = np.repeat(intercept, steps_enc)
        LR_deriv.append(part_slope)
        LR_bias.append(part_intercept)

    slope_data = np.ravel(np.array(LR_deriv)).reshape(-1, 1)
    intercept_data = np.ravel(np.array(LR_bias)).reshape(-1, 1)

    return slope_data,intercept_data


def lin_reg_train_test(LR_window, previous_data, current_data, border, batches, steps_enc, steps_dec):
    """
    the function computes the slopes and intercepts for the training and test dataset
    """
    LR_deriv = []
    LR_bias = []
    for part in range(batches):
        current_start = border + part*steps_dec
        if (current_start + steps_enc) < LR_window:
            before_window = LR_window - current_start - steps_enc
            before = previous_data[-before_window:]
            current = current_data[:current_start + steps_enc]
            for_LR = np.concatenate((before, current), axis=0)
        else:
            for_LR = current_data[current_start -
                                    (LR_window - steps_enc):current_start + steps_enc]
        X = range(len(for_LR))
        X = np.array(X)
        X = X[:, None]

        reg = linear_model.LinearRegression().fit(X, for_LR)
        reg.score(X, for_LR)
        slope = reg.coef_
        intercept = reg.intercept_
        
        part_slope = np.repeat(slope, steps_enc)
        part_intercept = np.repeat(intercept, steps_enc)
        LR_deriv.append(part_slope)
        LR_bias.append(part_intercept)

    slope_data = np.ravel(np.array(LR_deriv)).reshape(-1, 1)
    intercept_data = np.ravel(np.array(LR_bias)).reshape(-1, 1)

    return slope_data,intercept_data


def mean_split_valid(mean_window, valid_scaled, border, batches, steps_enc, steps_dec):
    """
    the function computes the means for the validation dataset
    """
    means = []
    for part in range(batches):
        current_start = border + part*steps_dec
        if (current_start + steps_enc) < mean_window:
            for_mean = valid_scaled[:current_start + steps_enc]

        else:
            for_mean = valid_scaled[current_start -
                                    (mean_window - steps_enc):current_start + steps_enc]
        part_mean = np.mean(for_mean)
        part_means = np.repeat(part_mean, steps_enc)
        means.append(part_means)

    mean_data = np.ravel(np.array(means)).reshape(-1, 1)

    return mean_data

def mean_split_train_test(mean_window, previous_data, current_data, border, batches, steps_enc, steps_dec):
    """
    the function computes the means for the training and test dataset
    """
    means = []
    # valid_border_scaled = np.concatenate(
    #     (valid_scaled, train_scaled.iloc[:border]), axis=0)
    for part in range(batches):
        current_start = border + part*steps_dec
        if (current_start + steps_enc) < mean_window:
            before_window = mean_window - current_start - steps_enc
            before = previous_data[-before_window:]
            current = current_data[:current_start + steps_enc]
            for_mean = np.concatenate((before, current), axis=0)
        else:
            for_mean = current_data[current_start -
                                    (mean_window - steps_enc):current_start + steps_enc]
        part_mean = np.mean(for_mean)
        part_means = np.repeat(part_mean, steps_enc)
        means.append(part_means)

    mean_data = np.ravel(np.array(means)).reshape(-1, 1)

    return mean_data


def normalization(valid_data, train_data, test_data, column):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_data[column].values.reshape(
        -1, 1))
    train_scaled = scaler.transform(train_data[column].values.reshape(
        -1, 1))
    test_scaled = scaler.transform(
        test_data[column].values.reshape(-1, 1))
    valid_scaled = scaler.transform(
        valid_data[column].values.reshape(-1, 1))

    return valid_scaled, train_scaled, test_scaled, scaler


def inverse_normalization(scaler, data):
    inverted_data = scaler.inverse_transform(data)
    return inverted_data


def zscore(valid_data, train_data, test_data):
    # we compute the mean and standard deviation of the train data and use the same for test data
    mean = np.mean(train_data)
    deviation = np.std(train_data, ddof=1) #ddof to have sample std
    train_scaled = (train_data - mean)/deviation
    test_scaled = (test_data - mean)/deviation
    valid_scaled = (valid_data - mean)/deviation

    train_scaled_mean = np.mean(train_scaled)

    return deviation, mean, valid_scaled, train_scaled, test_scaled, train_scaled_mean


def inverse_zscore(data_scaled, mean, deviation):
    data_scaled *= deviation
    data_scaled += mean
    data_rescaled = data_scaled

    return data_rescaled

def next_batch(data_prev, data, options):
    batch_whole = []
    valid_train_scaled = np.concatenate((data_prev, data), axis=0)
    if options['means'] == True:
        for i in range(options['batch_size']):
            rand_start = np.random.randint(
                0, len(data)-(options['steps_enc']+options['steps_dec']))
            batch = (data[rand_start:rand_start +
                        (options['steps_enc']+options['steps_dec'])]).reshape(-1,1)
            for_mean = valid_train_scaled[rand_start  + len(data_prev) - 
                                        (options['mean_part'] - options['steps_enc']):rand_start + len(data_prev) + options['steps_enc']]
            part_mean = np.mean(for_mean)
            mean_batch = (np.repeat(
               part_mean, options['steps_enc'] + options['steps_dec'])).reshape(-1, 1)
            batch = np.concatenate((batch, mean_batch), axis=1)
            batch_whole.append(batch)
        dim = 2
    elif options['LR'] == True:
        for i in range(options['batch_size']):
            rand_start = np.random.randint(
                0, len(data)-(options['steps_enc']+options['steps_dec']))
            batch = (data[rand_start:rand_start +
                        (options['steps_enc']+options['steps_dec'])]).reshape(-1,1)
            for_LR = valid_train_scaled[rand_start + len(data_prev) -
                                        (options['mean_part'] - options['steps_enc']):rand_start + len(data_prev) + options['steps_enc']]
            X = range(len(for_LR))
            X = np.array(X)
            X = X[:, None]

            reg = linear_model.LinearRegression().fit(X, for_LR)
            reg.score(X, for_LR)
            slope = reg.coef_
            intercept = reg.intercept_
                
            part_slope = (np.repeat(slope, options['steps_enc'] + options['steps_dec'])).reshape(-1, 1)
            part_intercept = (np.repeat(intercept, options['steps_enc'] + options['steps_dec'])).reshape(-1, 1)
            batch = np.concatenate((batch,part_slope,part_intercept),axis=1)
            batch_whole.append(batch)
        dim=3
    else:
        for i in range(options['batch_size']):
            rand_start = np.random.randint(
                0, len(data)-(options['steps_enc']+options['steps_dec']))
            batch = (data[rand_start:rand_start +
                        (options['steps_enc']+options['steps_dec'])]).reshape(-1,1)
            batch_whole.append(batch)
        dim = 1
    batch_whole = np.array(batch_whole).reshape(
        options['batch_size'], (options['steps_enc']+options['steps_dec']), dim)

    x = batch_whole[:, :options['steps_enc'], :]
    y = batch_whole[:, options['steps_enc']:, 0]
    y = y.reshape(-1, options['steps_dec'], 1)

    return x, y


def batch_whole_vt(data, steps_enc, steps_dec):
    if ((len(data)//steps_enc)*steps_enc - (steps_enc-steps_dec) - 2*steps_enc + steps_dec) % steps_dec == 0:
        data = data
        border = len(data) - ((len(data)-steps_dec) //
                              steps_enc)*steps_enc - steps_dec
        X = data[border:-steps_dec]
        if steps_enc == steps_dec:
            Y = data[steps_enc:, 0]
        else:
            Y = data[border + steps_enc:, 0]

    else:
        data = data[len(data) - (int((len(data) - 2 * steps_enc) /
                                     (steps_enc*steps_dec)) * steps_enc * steps_dec + 2*steps_enc):]
        if steps_enc == steps_dec:
            Y = data[steps_enc:, 0]
        else:
            Y = data[2*steps_enc - steps_dec:, 0]
        X = data[steps_enc - steps_dec:-steps_dec]
    Y_batch = Y.reshape(-1, steps_dec, 1)
    X_batch = []
    for batch in range(len(Y_batch)):
        Xbatch = X[batch*steps_dec:(batch+1) *
                   steps_dec + (steps_enc-steps_dec)]
        X_batch.extend(Xbatch)
    X_batch = np.array(X_batch).reshape(-1, steps_enc, 1)

    return Y, X_batch, Y_batch


def batch_whole_test(data, steps_enc, steps_dec):
    if ((len(data)//steps_enc)*steps_enc - (steps_enc-steps_dec) - 2*steps_enc + steps_dec) % steps_dec == 0:
        data = data
        border = len(data) - int((len(data)-steps_dec)/
                              steps_enc)*steps_enc - steps_dec
        X = data[:- steps_dec]
        if border == 0:
            if steps_enc == steps_dec:
                Y = data[steps_enc:]
            else:
                Y = data[steps_enc:]
        else:
            if steps_enc == steps_dec:
                Y = data[steps_enc:-border]
            else:
                Y = data[steps_enc:-border]
    else:
        data = data[:int((len(data) - 2 * steps_enc) /
                         (steps_enc*steps_dec)) * steps_enc * steps_dec + 2*steps_enc]
        border = 0
        X = data[:- steps_enc]
        if steps_enc == steps_dec:
            Y = data[steps_enc:]
        else:
            Y = data[steps_enc:-(steps_enc-steps_dec)]
    Y_batch = Y.reshape(-1, steps_dec, 1)
    X_batch = []
    for batch in range(len(Y_batch)):
        Xbatch = X[batch*steps_dec:(batch+1) *
                   steps_dec + (steps_enc-steps_dec)]
        X_batch.extend(Xbatch)
    X_batch = np.array(X_batch).reshape(-1, steps_enc, 1)

    return Y, X_batch, Y_batch


##########################  Visualization  #############################

def subplot(Y_train, Y_train_pred, Y_test, Y_test_pred, mse_train, mse_test, path, iteration):
    plt.figure(figsize=(15, 10))
    plt.suptitle('Iteration ' + str(iteration) + ' train MSE: ' +
                 str(mse_train) + ' test MSE: ' + str(mse_test))
    plt.subplot(211)
    plt.plot(Y_train, 'g-', label='Train target')
    plt.plot(Y_train_pred, 'b-', label='Train predicton')
    plt.xlabel('Time')
    plt.ylabel('Response variable')
    plt.legend()

    plt.subplot(212)
    plt.plot(Y_test, 'g-', label='Test target')
    plt.plot(Y_test_pred, 'b-', label='Test prediction')
    plt.xlabel('Time')
    plt.ylabel('Response variable')
    plt.legend()

    file_name = path + '/' + 'prediction_' + str(iteration) + '.png'
    plt.savefig(file_name)
    plt.close()


def final_subplot(Y_train, Y_train_pred, Y_test, Y_test_pred, mse_train, mse_test, rmse_train, rmse_test,path, iteration):
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = fig.add_subplot(211)
    ax1.title.set_text('Iteration' + str(iteration) + ' - train MSE: ' + str(mse_train) + ', RMSE: ' + str(rmse_train))
    plt.plot(Y_train, 'g-', label='Train target')
    plt.plot(Y_train_pred, 'b-', label='Train predicton')
    plt.xlabel('Time')
    plt.ylabel('Response variable')
    plt.legend()

    ax2 = fig.add_subplot(212)
    ax2.title.set_text('Iteration' + str(iteration) + ' - test MSE: ' + str(mse_test) + ', RMSE: ' + str(rmse_test))
    plt.plot(Y_test, 'g-', label='Test target')
    plt.plot(Y_test_pred, 'b-', label='Test prediction')
    plt.xlabel('Time')
    plt.ylabel('Response variable')
    plt.legend()

    file_name = path + '/' + 'final_prediction_' + str(iteration) + '.png'
    plt.savefig(file_name)
    plt.close()


def plot_parts(num_time_steps, data, pred, name, path):
    sequences_in_plot = 10
    if num_time_steps < 10:
        sequences_in_plot = 200
    parts = int(np.ceil(len(data)/(sequences_in_plot*num_time_steps)))

    rmse = []
    for part in range(parts):
        # if part == parts-1:
        #     part_length = len(data) - (parts-1) * \
        #         hp.num_time_steps*sequences_in_plot
        # else:
        part_length = num_time_steps*sequences_in_plot

        rmse_part = np.sqrt(np.mean(
            (data[part*part_length:(part+1)*part_length]-pred[part*part_length:(part+1)*part_length])**2))
        mean_part = np.mean(data[part*part_length:(part+1)*part_length])

        lines = [
            i*num_time_steps for i in range(int(len(pred[part*part_length:(part+1)*part_length])/num_time_steps))]
        minmax = [min(min(pred[part*part_length:(part+1)*part_length]), min(data[part*part_length:(part+1)*part_length])),
                  max(max(pred[part*part_length:(part+1)*part_length]), max(data[part*part_length:(part+1)*part_length]))]
        plt.figure(figsize=(15, 5))
        plt.plot(data[part*part_length:(part+1)*part_length],
                 'g-', label='Original data')
        plt.plot(pred[part*part_length:(part+1)*part_length],
                 'b-', label='Prediction')
        plt.vlines(lines, minmax[0], minmax[1],
                   colors='gray', linestyles='dotted')
        plt.title(name + ' - RMSE: ' + str(rmse_part) +
                  ', part mean: ' + str(mean_part))
        plt.xlabel('Time')
        plt.ylabel('Response variable')
        plt.legend()
        file_name = path + '/' + name + '_' + str(part+1) + '.png'
        plt.savefig(file_name)
        plt.close()

        rmse.append(rmse_part)
    return rmse


################################# Saving data #####################################
def make_hparam_string(options):
    # from https://github.com/llSourcell/how_to_use_tensorboard_live/blob/master/mnist.py
    return 'd_%s,m_%s,c_%s,we_%s,wd_%s,bs_%s,nl_%s,nh_%s' % (options['input_data'], options['model_type'], options['cell_type'], options['steps_enc'], options['steps_dec'], options['batch_size'], options['num_layers'], options['num_hidden'])



def make_path(options):
    hparam = make_hparam_string(options)
    logdir = './tmp/foruse/'
    path = logdir + hparam
    path_index = 1
    if options['LR'] == True:
        while os.path.exists(path + '_LR_' + str(options['LR_part']) + '_' + str(path_index)) == True:
            path_index += 1
        path = path + '_LR_' + str(options['LR_part']) + '_' + str(path_index)
    elif options['means'] == True:
        while os.path.exists(path + '_mean_' + str(options['mean_part']) + '_' + str(path_index)) == True:
            path_index += 1
        path = path + '_mean_' + str(options['mean_part']) + '_' + str(path_index)
    else:
        while os.path.exists(path + '_' + str(path_index)) == True:
            path_index += 1
        path = path + '_' + str(path_index)

    return path


def save_data(path, name, data):
    df = pd.DataFrame(data)
    outfile = open(path + '/' + name + '.csv', 'w')
    df.to_csv(outfile, index=None)
    outfile.close()



def save_hyperparameters(options,path):
    num_units = []
    for layer in range(options['num_layers']):
        num_units.append(options['num_hidden'])
    with open(os.path.join(path, 'Hyperparameters.txt'), 'w') as text_file:
        text_file.write('Input data: %s\r\n' % options['input_data'])
        text_file.write('Model type: %s\r\n' % options['model_type'])
        text_file.write('Optimizer: %s\r\n' % options['optimizer_type'])
        text_file.write('Cell type: %s\r\n' % options['cell_type'])
        text_file.write('Encoder window size: %d\r\n' % options['steps_enc'])
        text_file.write('Decoder window size: %d\r\n' % options['steps_dec'])
        text_file.write('Train part: %s\r\n' % str(options['train_ratio']))
        text_file.write('Learning rate: %s\r\n' % str(options['learning_rate']))
        text_file.write('Batch size: %d\r\n' % options['batch_size'])
        text_file.write('Hidden layers: %d\r\n' % len(num_units))
        for i in range(len(num_units)):
            text_file.write('Layer: %d\r\n' % num_units[i])
        if options['LR'] == True:
            text_file.write('LR parts: %d\r\n' % options['LR_part'])
        if options['means'] == True:
            text_file.write('Mean parts: %d\r\n' % options['mean_part'])
        text_file.write('Early stopping: %d\r\n' % options['max_patience'])
        text_file.write('Path: %s\r\n' % path )
        text_file.close()
