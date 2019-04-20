# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 00:22:41 2019

@author: Shalin
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
class Eval(object):
    def __init__(self):
        self.recall_list = []
        self.precision_list = []
        self.accuracy_list = []
        self.true_positive = 0
        self.positive_sample = 0
        self.predict_positive = 0
        self.correct_prediction = 0
        self.num_sample = 0

        self.f1_score = []

    def eval_session(self, label_path, predict_path):
        

        equal_list = (label_file == predict_file).astype(np.int)

        positive_sample = np.count_nonzero(label_file == 1.)
        negative_sample = np.count_nonzero(label_file == 0.)

        true_positive = np.count_nonzero(np.all((label_file == 1., predict_file == 1.), axis=0))
        true_negative = np.count_nonzero(np.all((label_file == 0., predict_file == 0.), axis=0))

        predict_positive = np.count_nonzero(predict_file == 1.)

        correct_prediction = np.count_nonzero(label_file == predict_file)
        num_sample = len(label_file)

        recall = get_recall(true_positive, positive_sample)
        precision = get_precision(true_positive, predict_positive)
        accuracy = correct_prediction / num_sample

        self.recall_list.append(recall)
        self.precision_list.append(precision)
        self.accuracy_list.append(accuracy)
        self.f1_score.append(get_f1(precision, recall))
        self.true_positive += true_positive
        self.positive_sample += positive_sample
        self.predict_positive += predict_positive
        self.correct_prediction += correct_prediction
        self.num_sample += num_sample

    def print_eval(self):
        print('=== Average over Session: ===')
        print('Recall: {}, Precision: {}, Accuracy: {}, F1: {}'.\
                format(np.mean(self.recall_list), np.mean(self.precision_list), 
                       np.mean(self.accuracy_list), np.mean(self.f1_score)))

        print('=== Entire Data: ===')
        recall = get_recall(self.true_positive, self.positive_sample)
        precision = get_precision(self.true_positive, self.predict_positive)
        accuracy = self.correct_prediction / self.num_sample
        f1 = get_f1(precision, recall)

        print('Recall: {}, Precision: {}, Accuracy: {}, F1: {}'.\
                format(recall, precision, accuracy, f1))

def get_f1(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        return 2. * (precision * recall) / (precision + recall)


def get_recall(true_positive, positive_sample):
    if positive_sample == 0:
        return 1.
    else:
        return true_positive / positive_sample

def get_precision(true_positive, predict_positive):
    if predict_positive == 0:
        return 1.
    else:
        return true_positive / predict_positive

if __name__ == '__main__':
    label_path = 'D:\\NCSU\\sem2\\ECE 542 Neural Networks\\project\\3b\\ECE542-Proj3b\\Training Data B'
    predict_file_name = 'prediction.txt'

    eval_obj = Eval()

    session_list = [13,15,16]
    label_file={}
    predict_file={}
    for session_id in session_list:
        cur_label_path = os.path.join(label_path, 'Session{:02d}'.format(session_id), 'detection.txt')
        cur_predict_path = os.path.join(label_path, 'Session{:02d}'.format(session_id), predict_file_name)
#        eval_obj.eval_session(cur_label_path, cur_predict_path)

#    eval_obj.print_eval()

        label_file[session_id] = np.loadtxt(cur_label_path) 
        predict_file[session_id] = np.loadtxt(cur_predict_path)
    for d,p in zip(label_file.items(),predict_file.items()):
        plt.figure()
        plt.title('Session'+str(d[0]))
        plt.xlabel('Time in  milliseconds')
        plt.ylabel('Event')
#        plt.xticks(np.arange(10000,15000,100),rotation='vertical')
        plt.plot(d[1][10000:15000],'ro',markersize=6)
        plt.plot(p[1][10000:15000],'bo',markersize=2)
        plt.legend(('Ground Truth','Prediction'),loc=7)
        plt.savefig(str(d[0])+'.jpg')
    plt.show()