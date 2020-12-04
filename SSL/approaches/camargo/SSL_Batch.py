import pandas as pd
import os
import itertools
from nltk.util import ngrams
import keras.utils as ku
import SSL_Embedding as embd
import SSL_Evaluate as ev
import numpy as np
import csv
from model_training import model_loader as mload
from support_modules import support as sup
from keras.models import load_model


def load_embedded(index, filename):
    """Loading of the embedded matrices.
    parms:
        index (dict): index of activities or roles.
        filename (str): filename of the matrix file.
    Returns:
        numpy array: array of weights.
    """
    weights = list()
    input_folder = os.path.join('input_files', 'embedded_matrix')
    with open(os.path.join(input_folder, filename), 'r') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in filereader:
            cat_ix = int(row[0])
            if index[cat_ix] == row[1].strip():
                weights.append([float(x) for x in row[2:]])
        csvfile.close()
    return np.array(weights)


BASE_DIRECTORY = "I:\Lab\Real_Life_Event_Logs"
EVENT_LOG = "Helpdesk"

parameters = dict()
parameters['imp'] = 2  # keras lstm implementation 1 cpu,2 gpu
parameters['lstm_act'] = 'relu'  # optimization function Keras
parameters['dense_act'] = None  # optimization function Keras
parameters['optim'] = 'Adam'  # optimization function Keras
parameters['norm_method'] = 'lognorm'  # max, lognorm # Model types --> shared_cat, shared_cat_inter,# seq2seq, seq2seq_inter
parameters['model_type'] = 'shared_cat'
parameters['n_size'] = 5  # n-gram size
parameters['l_size'] = 100  # LSTM layer sizes
#parameters['training_log'] = 'inp_log_train_red_10.csv'
parameters['test_log'] = 'inp_log_test.csv'


PATH_VARIANT = "data\SplitterByStartingTime\\0.3\ReducerRandom\GenerativeLSTM"
activities_df = pd.read_csv(os.path.join(BASE_DIRECTORY, EVENT_LOG, PATH_VARIANT, 'inp_activities.csv'))
roles_df = pd.read_csv(os.path.join(BASE_DIRECTORY, EVENT_LOG, PATH_VARIANT, 'inp_roles.csv'))
pairs_df = pd.read_csv(os.path.join(BASE_DIRECTORY, EVENT_LOG, PATH_VARIANT, 'inp_pairs.csv'))

embeddings = False
training = False
test = True

if(embeddings == True):
    # =============================================================================
    # Embeddings
    # =============================================================================
    print("----------------------START TRAINING EMBEDDINGS-------------")
    ac_index, rl_index = embd.train_embedded(pairs_df, activities_df, roles_df, EVENT_LOG)
    index_ac = {v: k for k, v in ac_index.items()}
    index_rl = {v: k for k, v in rl_index.items()}

if(training == True):
    # =============================================================================
    # Training of the model
    # =============================================================================
    print("----------------------START TRAINING OF THE MODEL-------------")

    def train(path_trainings_log, event_log_name, id_key_for_log):
        # Laden der Trainingsmenge
        log_train = pd.read_csv(path_trainings_log, encoding='unicode_escape')
        ac_index = dict(activities_df.values.tolist())
        rl_index = dict(roles_df.values.tolist())

        index_ac = {v: k for k, v in ac_index.items()}
        index_rl = {v: k for k, v in rl_index.items()}

        equi = {'ac_index': 'activities',
                'rl_index': 'roles',
                'dur_norm': 'times'}
        columns = list(equi.keys())
        vec = {'prefixes': dict(),
               'next_evt': dict(),
               'max_dur': np.max(log_train.dur)}

        temp_data = list()
        log_df = log_train.to_dict('records')
        key = 'end_timestamp'
        log_df = sorted(log_df, key=lambda x: (x['caseid'], key))

        for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()

            for x in columns:
                serie = [y[x] for y in trace]
                if x == 'ac_index':
                    serie.insert(0, ac_index[('Start')])
                    serie.append(ac_index[('End')])
                elif x == 'rl_index':
                    serie.insert(0, rl_index[('Start')])
                    serie.append(rl_index[('End')])
                else:
                    serie.insert(0, 0)
                    serie.append(0)
                temp_dict = {**{x: serie}, **temp_dict}
            temp_dict = {**{'caseid': key}, **temp_dict}
            temp_data.append(temp_dict)

        # n-gram definition
        for i, _ in enumerate(temp_data):
            for x in columns:
                serie = list(ngrams(temp_data[i][x], parameters['n_size'],
                                    pad_left=True, left_pad_symbol=0))
                print("serie", i, x, serie)
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                # print("serie", i, x, serie)
                y_serie = y_serie[1:]
                # print("y_serie", i, x, y_serie)
                vec['prefixes'][equi[x]] = vec['prefixes'][equi[x]] + serie if i > 0 else serie
                vec['next_evt'][equi[x]] = vec['next_evt'][equi[x]] + y_serie if i > 0 else y_serie

                # Transform task, dur and role prefixes in vectors
        for value in equi.values():
            vec['prefixes'][value] = np.array(vec['prefixes'][value])
            vec['next_evt'][value] = np.array(vec['next_evt'][value])

        # Reshape dur (prefixes, n-gram size, 1) i.e. time distribute
        vec['prefixes']['times'] = vec['prefixes']['times'].reshape(
            (vec['prefixes']['times'].shape[0],
             vec['prefixes']['times'].shape[1], 1))
        # one-hot encode target values
        vec['next_evt']['activities'] = ku.to_categorical(
            vec['next_evt']['activities'], num_classes=len(ac_index))
        vec['next_evt']['roles'] = ku.to_categorical(
            vec['next_evt']['roles'], num_classes=len(rl_index))

        # Load embedded matrix
        ac_weights = load_embedded(index_ac, 'ac_' + event_log_name + '.emb')
        rl_weights = load_embedded(index_rl, 'rl_' + event_log_name + '.emb')

        folder_id = sup.folder_id()+id_key_for_log
        output_folder = os.path.join('output_files', folder_id)

        # Export params
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            os.makedirs(os.path.join(output_folder, 'parameters'))

        f = open(os.path.join(output_folder, 'description'), "w")
        f.write(path_trainings_log)
        f.write(event_log_name)
        f.write(id_key_for_log)
        f.close()

        parameters['index_ac'] = index_ac
        parameters['index_rl'] = index_rl

        if parameters['model_type'] in ['shared_cat', 'shared_cat_inter']:
            print("IF")
            parameters['dim'] = dict(
                samples=str(vec['prefixes']['activities'].shape[0]),
                time_dim=str(vec['prefixes']['activities'].shape[1]),
                features=str(len(ac_index)))
        else:
            parameters['dim'] = dict(
                samples=str(vec['encoder_input_data']['activities'].shape[0]),
                time_dim=str(vec['encoder_input_data']['activities'].shape[1]),
                features=str(len(ac_index)))
        parameters['max_dur'] = str(vec['max_dur'])

        sup.create_json(parameters, os.path.join(output_folder,
                                                 'parameters',
                                                 'model_parameters.json'))

        # Trainieren des Models
        m_loader = mload.ModelLoader(parameters)
        m_loader.train(parameters['model_type'], vec, ac_weights, rl_weights, output_folder)

    event_logs_to_train = [#'inp_log_train_red_10.csv',
                           #'inp_log_train_red_08.csv',
                           'inp_log_train_red_06.csv',
                           'inp_log_train_red_04.csv',
                           'inp_log_train_red_02.csv',
                           'inp_log_train_red_01.csv',
                           'inp_log_train_red_005.csv',
                           'inp_log_train_red_001.csv'
                           ]

    for i in event_logs_to_train:
        print("Train model for ", i)
        train(os.path.join(BASE_DIRECTORY, EVENT_LOG, PATH_VARIANT, i),
              EVENT_LOG,
              i)



# =============================================================================
# Evaluation of the Model
# =============================================================================
if(test == True):
    def evaluate_model(path_test_log, folder, model_file, parameters):
        log_test = pd.read_csv(path_test_log, encoding='unicode_escape')

        parameters['folder'] = folder
        parameters['model_file'] = model_file
        parameters['variants'] = [{'imp': 'Random Choice', 'rep': 1}, {'imp': 'Arg Max', 'rep': 1}]
        parameters['activity'] = 'predict_next'

        output_route = os.path.join('output_files', parameters['folder'])
        model = load_model(os.path.join(output_route, parameters['model_file']))

        import json

        path = os.path.join(output_route,
                            'parameters',
                            'model_parameters.json')
        with open(path) as file:
            data = json.load(file)
            if 'activity' in data:
                del data['activity']
            parameters = {**parameters, **{k: v for k, v in data.items()}}
            parameters['dim'] = {k: int(v) for k, v in data['dim'].items()}
            parameters['max_dur'] = float(data['max_dur'])
            parameters['index_ac'] = {int(k): v
                                      for k, v in data['index_ac'].items()}
            parameters['index_rl'] = {int(k): v
                                      for k, v in data['index_rl'].items()}
            file.close()
            ac_index = {v: k for k, v in parameters['index_ac'].items()}
            rl_index = {v: k for k, v in parameters['index_rl'].items()}

        ev.evaluate(log_test, ac_index, rl_index, parameters, model)


    print("----------------------START EVALUATION-------------")

    log_train_red_10 = {
        "folder": 'Helpdesk_SplitterByTime_ReducerRandom/20201203_134452142499inp_log_train_red_10.csv',
        "model_file": 'model_shared_cat_63-0.70.h5'
    }
    log_train_red_08 = {
        "folder": 'Helpdesk_SplitterByTime_ReducerRandom/20201203_144547320219inp_log_train_red_08.csv',
        "model_file": 'model_shared_cat_40-0.70.h5'
    }
    log_train_red_06 = {
        "folder": 'Helpdesk_SplitterByTime_ReducerRandom/20201203_152603540551inp_log_train_red_06.csv',
        "model_file": 'model_shared_cat_17-0.75.h5'
    }
    log_train_red_04 = {
        "folder": 'Helpdesk_SplitterByTime_ReducerRandom/20201203_154934419936inp_log_train_red_04.csv',
        "model_file": 'model_shared_cat_49-0.69.h5'
    }
    log_train_red_02 = {
        "folder": 'Helpdesk_SplitterByTime_ReducerRandom/20201203_161219442875inp_log_train_red_02.csv',
        "model_file": 'model_shared_cat_62-0.71.h5'
    }
    log_train_red_01 = {
        "folder": 'Helpdesk_SplitterByTime_ReducerRandom/20201203_162515066720inp_log_train_red_01.csv',
        "model_file": 'model_shared_cat_17-0.79.h5'
    }
    log_train_red_005 = {
        "folder": 'Helpdesk_SplitterByTime_ReducerRandom/20201203_162909528482inp_log_train_red_005.csv',
        "model_file": 'model_shared_cat_37-0.78.h5'
    }
    log_train_red_001 = {
        "folder": 'Helpdesk_SplitterByTime_ReducerRandom/20201203_163200642478inp_log_train_red_001.csv',
        "model_file": 'model_shared_cat_142-0.95.h5'
    }

    to_evaluate = [
        log_train_red_10,
        log_train_red_08,
        log_train_red_06,
        log_train_red_04,
        log_train_red_02,
        log_train_red_01,
        log_train_red_005,
        log_train_red_001
    ]

    for i in to_evaluate:
        print("Evaluiere: ", i)
        evaluate_model(os.path.join(BASE_DIRECTORY, EVENT_LOG, PATH_VARIANT, parameters['test_log']), i['folder'], i['model_file'], parameters)










