import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.utils import np_utils
from keras.layers import Conv2D, Activation
from keras import regularizers
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
import os



def dataset_summary(path):
    dataset_df = pd.read_csv(path, sep=",")
    expected_colums = {'CaseID', 'Activity', 'Timestamp'}
    if set(dataset_df.columns) == expected_colums :
        activity_distribution = dataset_df['Activity'].value_counts()
        n_caseid = dataset_df['CaseID'].nunique()
        n_activity = dataset_df['Activity'].nunique()
        cont_trace = dataset_df['CaseID'].value_counts(dropna=False)
        max_trace_length = max(cont_trace)
        min_trace_length = min(cont_trace)
        mean_trace_length = np.mean(cont_trace)

        n_events = dataset_df['Activity'].count()
        print("Activity Distribution:\n", activity_distribution)
        print("Number of CaseIDs: ", n_caseid)
        print("Number of Unique Activities: ", n_activity)
        print("Number of Activities: ", n_events)
        print("Max length trace: ", max_trace_length)
        print("Mean length trace: ", mean_trace_length)
        print("Min length trace: ", min_trace_length)
        return dataset_df, max_trace_length, n_caseid, n_activity
    else:
        print("Error: The input format does not match")
        return


def get_image(act_val, time_val, max_trace_length, n_activity):
    i = 0
    matrix_zero = [max_trace_length, n_activity, 2]
    image = np.zeros(matrix_zero)
    list_image = []
    # Iteriere ueber traces
    while i < len(time_val):
        j = 0
        list_act = []
        list_temp = []
        dictionary_cont = dict()
        for k in range(0, n_activity):
            dictionary_cont.update({k + 1: 0})
        dictionary_diff = dict()
        for k in range(0, n_activity):
            dictionary_diff.update({k + 1: 0})

        while j < (len(act_val.iat[i, 0]) - 1):
            start_trace = time_val.iat[i, 0][0]
            dictionary_cont[act_val.iat[i, 0][0 + j]] = dictionary_cont[act_val.iat[i, 0][0 + j]] + 1
            dictionary_diff[act_val.iat[i, 0][0 + j]] = time_val.iat[i, 0][0 + j] - start_trace

            temp_cond_list = []
            for key in dictionary_cont:
                temp_cond_list.append(dictionary_cont[key])
            list_act.append(temp_cond_list)

            temp_diff_list = []
            for key in dictionary_diff:
                temp_diff_list.append(dictionary_diff[key])
            list_temp.append(temp_diff_list)
            j = j + 1
            cont = 0
            lenk = len(list_act) - 1
            while cont <= lenk:
                for l in range(0, n_activity):
                    image[(max_trace_length - 1) - cont][l] = [list_act[lenk - cont][l], list_temp[lenk - cont][l]]
                cont = cont + 1
            if cont == 1:
                pass
            else:
                list_image.append(image)
                image = np.zeros(matrix_zero)
        i = i + 1
    return list_image


def get_label(act):
    i = 0
    list_label = []
    while i < len(act):
        j = 0
        while j < (len(act.iat[i, 0]) - 1):
            if j > 0:
                list_label.append(act.iat[i, 0][j + 1])
            else:
                pass
            j = j + 1
        i = i + 1
    return list_label


# =============================================================================
# Definition of the neural network architecture
# =============================================================================
def create_neural_network(max_trace_length, n_activity, num_classes):
    model = Sequential()
    reg = 0.0001
    # Input Layer
    input_shape = (max_trace_length, n_activity, 2)

    # Layer 1
    model.add(Conv2D(32, (2, 2), input_shape=input_shape, padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(64, (4, 4), padding='same', kernel_regularizer=regularizers.l2(reg), ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(Conv2D(128, (8, 8), padding='same', kernel_regularizer=regularizers.l2(reg), ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattining
    model.add(Flatten())

    # Output Layer
    model.add(Dense(num_classes, activation='softmax', name='act_output'))

    return model

def create_neural_network_bpic_12(max_trace_length, n_activity, num_classes):
    model = Sequential()
    reg = 0.0001
    input_shape = (max_trace_length, n_activity, 2)
    model.add(Conv2D(32, (2, 2), input_shape=input_shape, padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (4, 4), padding='same', kernel_regularizer=regularizers.l2(reg), ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', name='act_output'))

    return model

def main(path_reference_log, training_logs):
    dataset_reference_log, max_trace_length, n_caseid, n_activity = dataset_summary(path_reference_log)

    reference_act = dataset_reference_log.groupby('CaseID').agg({'Activity': lambda x: list(x)})

    l_reference = get_label(reference_act)
    le = preprocessing.LabelEncoder()
    l_reference = le.fit_transform(l_reference)

    for training_log in training_logs:
        training_log_df = pd.read_csv(training_log['training_file'], sep=",")
        test_log_df = pd.read_csv(training_log['test_file'], sep=",")

        training_act = training_log_df.groupby('CaseID').agg({'Activity': lambda x: list(x)})
        training_time = training_log_df.groupby('CaseID').agg({'Timestamp': lambda x: list(x)})

        test_act = test_log_df.groupby('CaseID').agg({'Activity': lambda x: list(x)})
        test_time = test_log_df.groupby('CaseID').agg({'Timestamp': lambda x: list(x)})

        # Create labeled data set of training
        X_training = get_image(training_act, training_time, max_trace_length, n_activity)
        X_test = get_image(test_act, test_time, max_trace_length, n_activity)

        l_training = get_label(training_act)
        l_test = get_label(test_act)

        l_training = le.transform(l_training)
        l_test = le.transform(l_test)

        X_training = np.asarray(X_training)
        l_training = np.asarray(l_training)

        # Create labeled data set for test
        X_test = np.asarray(X_test)
        l_test = np.asarray(l_test)

        train_Y_one_hot = np_utils.to_categorical(l_training, le.classes_.size)
        test_Y_one_hot = np_utils.to_categorical(l_test, le.classes_.size)

        # =============================================================================
        # Train neural network
        # =============================================================================
        model = create_neural_network(max_trace_length, n_activity, le.classes_.size)
        print(model.summary())

        # Configure the model for training
        opt = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
        model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt, metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=6)

        # Train the model
        history = model.fit(X_training, {'act_output': train_Y_one_hot}, validation_split=0.2, verbose=1,
                            callbacks=[early_stopping], batch_size=128, epochs=500)
        model.save(training_log['log_name'] + '_Ratio' + training_log['training_test_ratio'] + "_Splitter"+training_log['splitter']+"_Reducer_"+training_log['reducer']+"_"+training_log['reduction_factor']+".h5")

        # Evaluate
        path = os.path.join("I:/", training_log['log_name'] + '_Ratio' + training_log['training_test_ratio']+'_Splitter' + training_log['splitter'] + '_Reducer' +
                            training_log['reducer'] + '.txt')

        # Print confusion matrix for training data
        y_pred_train = model.predict(X_training)
        max_y_pred_train = np.argmax(y_pred_train, axis=1)
        score = model.evaluate(X_test, test_Y_one_hot, verbose=1)
        y_pred_test = model.predict(X_test)
        # Take the class with the highest probability from the test predictions
        max_y_pred_test = np.argmax(y_pred_test, axis=1)
        max_y_test = np.argmax(test_Y_one_hot, axis=1)

        # Take the class with the highest probability from the train predictions
        with open(path, "a") as evaluation_file:
            evaluation_file.write("Training File: " + training_log['training_file'] +
                                  "\n Training-Test-Ratio: " + training_log['training_test_ratio'] +
                                  "\n Splitter: " + training_log['splitter'] +
                                  "\n Reducer: " + training_log['reducer'] +
                                  "\n Reduction Factor: " + training_log['reduction_factor'] +
                                  "\n Test File: " + training_log['test_file'] + "\n")
            evaluation_file.write(classification_report(l_training, max_y_pred_train, digits=5))
            #evaluation_file.write(model.metrics_names)
            evaluation_file.write('\nAccuracy on test data: ' + str(score[1]))
            evaluation_file.write('\nLoss on test data: ' + str(score[0]))
            evaluation_file.write(classification_report(max_y_test, max_y_pred_test, digits=5))
            evaluation_file.write("\n------------------------------------------\n")


# Main program
log_train_red_001 = {
    "training_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_train_red_001.csv',
    "test_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_test.csv',
    "log_name": 'BPIC15_2',
    "reduction_factor": '99',
    "splitter": 'Time',
    "training_test_ratio": '03',
    "reducer": 'Time',
}

log_train_red_005 = {
    "training_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_train_red_005.csv',
    "test_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_test.csv',
    "log_name": 'BPIC15_2',
    "reduction_factor": '95',
    "splitter": 'Time',
    "training_test_ratio": '03',
    "reducer": 'Time',
}

log_train_red_01 = {
    "training_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_train_red_01.csv',
    "test_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_test.csv',
    "log_name": 'BPIC15_2',
    "reduction_factor": '90',
    "splitter": 'Time',
    "training_test_ratio": '03',
    "reducer": 'Time',
}

log_train_red_02 = {
    "training_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_train_red_02.csv',
    "test_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_test.csv',
    "log_name": 'BPIC15_2',
    "reduction_factor": '80',
    "splitter": 'Time',
    "training_test_ratio": '03',
    "reducer": 'Time',
}

log_train_red_04 = {
    "training_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_train_red_04.csv',
    "test_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_test.csv',
    "log_name": 'BPIC15_2',
    "reduction_factor": '60',
    "splitter": 'Time',
    "training_test_ratio": '03',
    "reducer": 'Time',
}

log_train_red_06 = {
    "training_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_train_red_06.csv',
    "test_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_test.csv',
    "log_name": 'BPIC15_2',
    "reduction_factor": '40',
    "splitter": 'Time',
    "training_test_ratio": '03',
    "reducer": 'Time',
}

log_train_red_08 = {
    "training_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_train_red_08.csv',
    "test_file": 'I:\Lab\Real_Life_Event_Logs\BPI_Challenge_2015_2\data\SplitterByStartingTime\\0.3\ReducerByTime\Pasquadibisceglie\inp_log_test.csv',
    "log_name": 'BPIC15_2',
    "reduction_factor": '20',
    "splitter": 'Time',
    "training_test_ratio": '03',
    "reducer": 'Time',
}

log_train_red_10 = {
    "training_file": 'dataset\inp_log_train_aug.csv',
    "test_file": 'dataset\inp_log_test.csv',
    "log_name": 'Helpdesk_Aug',
    "reduction_factor": '0',
    "splitter": 'Time',
    "training_test_ratio": '03',
    "reducer": 'Time',
}


training_logs = [log_train_red_10 ]

main("dataset\inp_referenceLog.csv", training_logs)

