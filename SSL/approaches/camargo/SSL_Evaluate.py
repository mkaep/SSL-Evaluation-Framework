import itertools

import os
import json

import pandas as pd
import numpy as np

from keras.models import load_model

from support_modules.readers import log_reader as lr
from support_modules import nn_support as nsup
from support_modules import support as sup

from model_prediction import interfaces as it
from model_prediction.analyzers import sim_evaluator as ev

# Reformat Events
def reformat_events(columns, log, ac_index, rl_index):
    temp_data = list()
    log_df = log.to_dict('records')
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
    return temp_data

def create_examples(log, ac_index, rl_index):
    columns = ['ac_index', 'rl_index', 'dur_norm']
    ref_log = reformat_events(columns, log, ac_index, rl_index)

    #Transformieren das verwendbar
    examples = {'prefixes': dict(), 'next_evt': dict()}

    # n-gram definition
    equi = {'ac_index': 'activities',
            'rl_index': 'roles',
            'dur_norm': 'times'}

    for i, _ in enumerate(ref_log):
        for x in columns:
            serie = [ref_log[i][x][:idx]
                     for idx in range(1, len(ref_log[i][x]))]
            y_serie = [x[-1] for x in serie]
            serie = serie[:-1]
            y_serie = y_serie[1:]
            examples['prefixes'][equi[x]] = (
                examples['prefixes'][equi[x]] + serie
                if i > 0 else serie)
            examples['next_evt'][equi[x]] = (
                examples['next_evt'][equi[x]] + y_serie
                if i > 0 else y_serie)

    return examples

def _predict_next_event_shared_cat(examples, model, imp, parameters):
    # Generation of predictions
    results = list()
    for i, _ in enumerate(examples['prefixes']['activities']):
        # Activities and roles input shape(1,5)
        x_ac_ngram = np.append(
                np.zeros(parameters['dim']['time_dim']),
                np.array(examples['prefixes']['activities'][i]), axis=0)[-parameters['dim']['time_dim']:].reshape((1, parameters['dim']['time_dim']))

        x_rl_ngram = np.append(
                np.zeros(parameters['dim']['time_dim']),
                np.array(examples['prefixes']['roles'][i]),
                axis=0)[-parameters['dim']['time_dim']:].reshape((1, parameters['dim']['time_dim']))

        # times input shape(1,5,1)
        x_t_ngram = np.array([np.append(
                np.zeros(parameters['dim']['time_dim']),
                np.array(examples['prefixes']['times'][i]),
                axis=0)[-parameters['dim']['time_dim']:].reshape((parameters['dim']['time_dim'], 1))])
        # add intercase features if necessary
        if parameters['model_type'] == 'shared_cat':
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
        elif parameters['model_type'] == 'shared_cat_inter':
            # times input shape(1,5,1)
            inter_attr_num = examples['prefixes']['inter_attr'][i].shape[1]
            x_inter_ngram = np.array([np.append(
                    np.zeros((parameters['dim']['time_dim'], inter_attr_num)),
                    examples['prefixes']['inter_attr'][i],
                    axis=0)[-parameters['dim']['time_dim']:].reshape((parameters['dim']['time_dim'], inter_attr_num))])
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
        # predict
        predictions = model.predict(inputs)
        if imp == 'Random Choice':
            # Use this to get a random choice following as PDF
            pos = np.random.choice(np.arange(0, len(predictions[0][0])),
                                    p=predictions[0][0])
            pos1 = np.random.choice(np.arange(0, len(predictions[1][0])),
                                    p=predictions[1][0])
        elif imp == 'Arg Max':
            # Use this to get the max prediction
            pos = np.argmax(predictions[0][0])
            pos1 = np.argmax(predictions[1][0])
        # save results
        results.append({
                'ac_prefix': examples['prefixes']['activities'][i],
                'ac_expect': examples['next_evt']['activities'][i],
                'ac_pred': pos,
                'rl_prefix': examples['prefixes']['roles'][i],
                'rl_expect': examples['next_evt']['roles'][i],
                'rl_pred': pos1})

    return results






def evaluate(log, ac_index, rl_index, parameters, model):
    predictions = None
    examples = create_examples(log, ac_index, rl_index)
    run_num = 0
    for variant in parameters['variants']:
        imp = variant['imp']
        run_num = 0
        for i in range(0, variant['rep']):
            results = pd.DataFrame(_predict_next_event_shared_cat(examples, model, imp, parameters))
            results['run_num'] = run_num
            results['implementation'] = imp
            if predictions is None:
                predictions = results
            else:
                predictions = predictions.append(results,
                                                           ignore_index=True)
            run_num += 1

    evaluator = EvaluateTask()
    evaluator.evaluate(parameters, predictions)


class EvaluateTask():

    def evaluate(self, parms, data):
        sampler = self._get_evaluator(parms['activity'])
        return sampler(data, parms)

    def _get_evaluator(self, activity):
        if activity == 'predict_next':
            return self._evaluate_predict_next
        else:
            raise ValueError(activity)

    def _evaluate_predict_next(self, data, parms):
        #exp_desc = self.clean_parameters(parms.copy())
        #print(data)
        evaluator = ev.Evaluator()
        ac_sim = evaluator.measure('accuracy', data, 'ac')
        rl_sim = evaluator.measure('accuracy', data, 'rl')
        # tm_mae = evaluator.measure('mae_suffix', data, 'tm')
        #exp_desc = pd.DataFrame([exp_desc])
        #exp_desc = pd.concat([exp_desc]*len(ac_sim), ignore_index=True)
        #ac_sim = pd.concat([ac_sim, exp_desc], axis=1).to_dict('records')
        #rl_sim = pd.concat([rl_sim, exp_desc], axis=1).to_dict('records')
        # tm_mae = pd.concat([tm_mae, exp_desc], axis=1).to_dict('records')
        print("AC: ", ac_sim)
        print("RL: ", rl_sim)
        #self.save_results(ac_sim, 'ac', parms)
        #self.save_results(rl_sim, 'rl', parms)
        # self.save_results(tm_mae, 'tm', parms)

    def clean_parameters(parms):
        exp_desc = parms.copy()
        exp_desc.pop('activity', None)
        exp_desc.pop('read_options', None)
        exp_desc.pop('column_names', None)
        exp_desc.pop('one_timestamp', None)
        exp_desc.pop('reorder', None)
        exp_desc.pop('index_ac', None)
        exp_desc.pop('index_rl', None)
        exp_desc.pop('dim', None)
        exp_desc.pop('max_dur', None)
        exp_desc.pop('variants', None)
        exp_desc.pop('is_single_exec', None)
        return exp_desc


    @staticmethod
    def save_results(measurements, feature, parms):
        if measurements:
            if parms['is_single_exec']:
                output_route = os.path.join('output_files',
                                            parms['folder'],
                                            'results')
                model_name, _ = os.path.splitext(parms['model_file'])
                sup.create_csv_file_header(
                    measurements,
                    os.path.join(
                        output_route,
                        model_name+'_'+feature+'_'+parms['activity']+'.csv'))
            else:
                if os.path.exists(os.path.join(
                        'output_files', feature+'_'+parms['activity']+'.csv')):
                    sup.create_csv_file(
                        measurements,
                        os.path.join('output_files',
                                     feature+'_'+parms['activity']+'.csv'),
                        mode='a')
                else:
                    sup.create_csv_file_header(
                        measurements,
                        os.path.join('output_files',
                                     feature+'_'+parms['activity']+'.csv'))
