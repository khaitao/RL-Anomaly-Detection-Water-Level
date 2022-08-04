from env import *
from lib import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import collections
import time

def main():
    DATA_DIR = os.path.abspath("./data/")

    SAVE_PATH = os.path.abspath("./save_model/")
    NUM_ACTIONS = 2 # number of valid actions (Normal, Anomalies)
    GAMMA = 0.99 # decay rate of past observations
    INITIAL_EPSILON = 0.1 # starting value of epsilon
    FINAL_EPSILON = 0.0001 # final value of epsilon
    MEMORY_SIZE = 50000 # number of previous transitions to remember

    # use survey is 0.5 % of total epochs
    NUM_EPOCHS_OBSERVE = 5
    NUM_EPOCHS_TRAIN = 100

    BATCH_SIZE = 256

    NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN

    REWARDS_CORRECT = 0.1
    REWARDS_CORRECT_ANOMALY = 0.9
    REWARDS_INCORRECT = -0.1

    #past_history = 12
    past_history = 6
    code= "Station1"
    start_date='2016-05-01 00:00:00'
    end_date='2016-06-30 23:50:00'

    env = EnvTimeSeries(code,DATA_DIR,past_history,NUM_ACTIONS,start_date, end_date, REWARDS_CORRECT, REWARDS_CORRECT_ANOMALY,REWARDS_INCORRECT)
    #BATCH_SIZE = len(env._y_train_batch)
    opt = Adam(lr=0.001)

    model = Sequential()
    model.add(Dense(36, input_shape=(past_history,), activation='relu'))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(2))
    # compile the keras model
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    model.summary()

    experience = collections.deque(maxlen=MEMORY_SIZE)
    modelName = "{}-GTDQN-MLP-{}Past-{}B{}EP-Inf-to{}".format(code, past_history, BATCH_SIZE, start_date[:10], end_date[:10])
    modelMinValid = "{}-GTDQN-MLP-MinValid-{}Past-{}B{}EP-{}-to{}".format(code,past_history, BATCH_SIZE, NUM_EPOCHS, start_date[:10], end_date[:10])
    modelMaxRwd = "{}-GTDQN-MLP-MaxReward-{}Past-{}B{}EP-{}-to{}".format(code, past_history, BATCH_SIZE, NUM_EPOCHS, start_date[:10], end_date[:10])
    modelMaxAcc = "{}-GTDQN-MLP-MaxAcc-{}Past-{}B{}EP-{}-to{}".format(code, past_history, BATCH_SIZE, NUM_EPOCHS, start_date[:10], end_date[:10])
    modelMaxF1 = "{}-GTDQN-MLP-MaxF1-{}Past-{}B{}EP-{}-to{}".format(code, past_history, BATCH_SIZE, NUM_EPOCHS, start_date[:10], end_date[:10])
    #LineNoti.LineNoti("Model {}@UEAPC Start Run".format(modelName))
    #fout = open(os.path.join(SAVE_PATH, "rl-network-results.tsv"), "wb")
    fout = open(os.path.join(SAVE_PATH, modelName+".tsv"), "w+")
    fout.write("GTDQN-MLP Run with Batch Size:{} Epochs:{}\n".format(BATCH_SIZE, NUM_EPOCHS))
    fout.write("Epocs\tTrainLoss\tValidLoss\tNumWin\tRunTime\tTP\tFN\tFP\tTN\tSpecificity\tRecall\tPrecision\tF1\n")

    epsilon = INITIAL_EPSILON
    avg_loss=100;
    e = 0
    check_abnormal = 0
    same_reward = 0
    prev_reward = 0.0
    train_loss = 100
    valid_loss = 100
    pre_val_loss = 100
    pre_train_loss = 100
    train_val_diff = 0.0
    pre_diff_val_train = 100

    min_val_loss = 100
    max_val_acc = 0
    max_f1 = 0

    pre_diff_divert = 0

    max_reward = 0

    for e in range(NUM_EPOCHS):
        loss = 0.0
        c_loss = 0
        episode_rwd = 0
        env.reset()
        popu_time = time.time()
        # get first state
        a_0 = 0
        s_t, r_0, game_over = env.step(a_0)
        while not game_over:
            s_tm1 = s_t
            # next action
            if e < NUM_EPOCHS_OBSERVE:
                a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
            else:
                if np.random.rand() <= epsilon:
                    a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
                else:
                    q = model.predict_on_batch(np.reshape(env._X_train_batch[env._states],(1, past_history)))
                    a_t = np.argmax(q[0])

            # apply action, get reward
            s_t, r_t, game_over = env.step(a_t)

            episode_rwd +=r_t
            # store experience
            experience.append((s_tm1, a_t, r_t, s_t, game_over))

            #X, Y = get_next_batch(experience, model, NUM_ACTIONS, GAMMA, BATCH_SIZE, env, past_history)

            if e >= NUM_EPOCHS_OBSERVE:
                # finished observing, now start training
                # get next batch
                X, Y = get_next_batch(experience, model, NUM_ACTIONS,
                                      GAMMA, BATCH_SIZE, env, past_history)

                #val_loss = model.train_on_batch(X, Y)[0]
                val_loss = model.train_on_batch(np.reshape(X,(-1, past_history)), Y)[0]
                c_loss += 1

                loss += val_loss
        #break
        train_loss, train_acc = model.evaluate(np.reshape(env._X_train_batch,(-1, past_history)), env._y_train_batch, verbose=0)
        valid_loss, valid_acc = model.evaluate(np.reshape(env._X_val_batch,(-1, past_history)), env._y_val_batch, verbose=0)

        # reduce epsilon gradually
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPOCHS

        if (c_loss > 0):
            avg_loss = loss/c_loss
        else:
            avg_loss = 100

        popu_time = time.time() - popu_time

        print("Code:{} Epoch {:04d}/{:d} | run time {:.2f} ".format(code, e + 1, NUM_EPOCHS, popu_time))
        print("validation loss:{:.4f} | Total Rwd:{:.4f}".format(valid_acc, episode_rwd))
        TP, FN, FP, TN, Spec, Rec, Prec, f1 = ModelAcc(env, model, past_history)
        print("========================================")

        fout.write("{:04d}\t{:.5f}\t{:.5f}\t{:.2f}\t{:.2f}\t{:d}\t{:d}\t{:d}\t{:d}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\n".format(e + 1, train_loss, valid_loss, episode_rwd, popu_time,TP, FN, FP, TN,Spec, Rec, Prec, f1))

        if e % 5 == 0:
            #LineNoti.LineNoti(lineTxt)
            model.save(os.path.join(SAVE_PATH, modelName+".h5"), overwrite=True)
            #print("Epoch {:04d}/{:d} | Loss {:.5f} | Win Count: {:d}".format(e + 1, NUM_EPOCHS, loss, num_wins))
        e += 1

        if (valid_loss < min_val_loss):  # save model when loss decreased
            model.save(os.path.join(SAVE_PATH, modelMinValid+".h5"), overwrite=True)
            min_val_loss = valid_loss

        if (valid_acc > max_val_acc):  # save model when loss decreased
            model.save(os.path.join(SAVE_PATH, modelMaxAcc+".h5"), overwrite=True)
            max_val_acc = valid_acc

        if (episode_rwd > max_reward):  # save model when loss decreased
            model.save(os.path.join(SAVE_PATH, modelMaxRwd +".h5"), overwrite=True)
            max_reward = episode_rwd

        if (f1 > max_f1):  # save model when loss decreased
            model.save(os.path.join(SAVE_PATH, modelMaxF1+".h5"), overwrite=True)
            max_f1 = f1
    #check_out.close()
    fout.close()
    model.save(os.path.join(SAVE_PATH, modelName+".h5"), overwrite=True)

if __name__ == '__main__':
	main()
