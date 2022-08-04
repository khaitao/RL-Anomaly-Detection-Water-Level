import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import random
import math
#from tensorflow.keras.utils import to_categorical

def get_next_batch(experience, model, num_actions, gamma, batch_size, _env, past_hist):
    sigmoid_v = np.vectorize(self_sigmoid)
    samples = random.sample(experience, batch_size)
    states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

    targets_batch = model.predict_on_batch(np.reshape(_env._X_train_batch[states_batch],(len(states_batch), past_hist)))
    future_qs = model.predict_on_batch(np.reshape(_env._X_train_batch[next_states_batch],(len(next_states_batch), past_hist)))

    Q_sa = np.argmax(future_qs, axis=-1)
    if gamma > 0:
        x = reward_batch+(gamma * Q_sa)
        x_sigmoid = sigmoid_v(x)
        for num, name in enumerate(targets_batch):
            targets_batch[num][action_batch[num]] = x_sigmoid[num]
    else:
        targets_batch = reward_batch

    return _env._X_train_batch[states_batch],targets_batch

def self_sigmoid(x):
    return 1 / (1 + math.exp(-x))

def transform_data_LastVal(dataset, predicted, start_index, end_index, history_size, target_size, train_mean, train_std):
    data = []
    labels = []

    data_normalized = (dataset-train_mean)/train_std

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size+1

    mid = history_size//2+1
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data.append(data_normalized[indices])
        labels.append(predicted[i-1])


    labels = tf.keras.utils.to_categorical(labels, num_classes=2)
    return np.array(data), np.array(labels)

def ModelAcc(env_, model, _past_history):
    q = model.predict(np.reshape(env_._X_val_batch,(-1, _past_history)))
    #print(q)
    anomaly = np.argmax(q, axis=-1)

    env_._val_df['anomalies'] = np.nan

    env_._val_df['anomalies'][_past_history-1:] = anomaly
    env_._val_df['GTDQN_Class'] = "Normal"
    env_._val_df['GTDQN_Class'].loc[env_._val_df['anomalies']==1] = "Anomaly"

    tn, fp, fn, tp = confusion_matrix(env_._val_df['GroundTruth_Class'], env_._val_df['GTDQN_Class'], labels=["Normal", "Anomaly"]).ravel()
    print("TP:{:d} | FN:{:d} | FP:{:d} | TN:{:d}".format(tp, fn, fp, tn))
    Specificity = tn/(tn+fp)
    Recall = tp/(tp+fn)
    Precision = tp/(tp+fp)
    F1 = 2*(Precision*Recall)/(Precision+Recall)
    #print("F1:{}".format(F1))
    print("Specificity:{:.4f} | Recall:{:.4f} | Precision:{:.4f} | F1:{:.4f}".format(Specificity, Recall, Precision, F1))
    return tp, fn, fp, tn,Specificity, Recall, Precision, F1
