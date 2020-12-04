import load_dataset as load
import LSTM_sequence_mae as ml
import datetime as date

#ml.train(150, 2, "data/referenceLog.csv", "data/log_train_red_10.csv", "data/log_test.csv") Root Mean Squared Error: 11.8300 d MAE: 9.5626 d MAPE: 3774.9670%
# ml.train(150, 2, "data/referenceLog.csv", "data/log_train_red_09.csv", "data/log_test.csv") Root Mean Squared Error: 12.4713 d MAE: 10.1570 d MAPE: 7241.5972%
#ml.train(150, 2, "data/referenceLog.csv", "data/log_train_red_08.csv", "data/log_test.csv") Root Mean Squared Error: 12.6787 d MAE: 10.3774 d MAPE: 7086.4097%
#ml.train(150, 2, "data/referenceLog.csv", "data/log_train_red_07.csv", "data/log_test.csv") Root Mean Squared Error: 12.0719 d MAE: 9.7128 d MAPE: 5540.6626%
#ml.train(150, 2, "data/referenceLog.csv", "data/log_train_red_06.csv", "data/log_test.csv") Root Mean Squared Error: 12.7023 d MAE: 10.2964 d MAPE: 5829.7500%
# ml.train(150, 2, "data/referenceLog.csv", "data/log_train_red_05.csv", "data/log_test.csv") Root Mean Squared Error: 12.2464 d MAE: 9.9528 d MAPE: 6724.8066% 220 Epcoehj
# ml.train(150, 2, "data/referenceLog.csv", "data/log_train_red_04.csv", "data/log_test.csv") Root Mean Squared Error: 12.8954 d MAE: 10.6230 d MAPE: 6296.7720% 318 Epochen, 43 Minuten
# ml.train(150, 2, "data/referenceLog.csv", "data/log_train_red_03.csv", "data/log_test.csv") Root Mean Squared Error: 12.3835 d MAE: 10.0883 d MAPE: 3689.2485% 35 Minuten, 341 Epcoehn
# ml.train(150, 2, "data/referenceLog.csv", "data/log_train_red_02.csv", "data/log_test.csv") 30 Minuten 424 Epochen Root Mean Squared Error: 12.5153 d MAE: 10.1550 d MAPE: 5999.6030%

#with open("data/result.txt", "a") as resultFile:
#    resultFile.write("Log Train Red 01: \n")
#    resultFile.write("Start: " + str(date.datetime.now()) + "\n")
#ml.train(150, 2, "data/referenceLog.csv", "data/log_train_red_01.csv", "data/log_test.csv", "data/result.txt", "Helpdesk_01")

#with open("data/result.txt", "a") as resultFile:
#    resultFile.write("End: " + str(date.datetime.now()) + "\n")


#####SplitterByStartingTime\0.3\ReducerRandom\Navarin

with open("data/BPIC_12/BPIC_12_SplitterTime_02_Random.txt", "a") as resultFile:
    resultFile.write("Log Train Red 10: \n")
    resultFile.write("Start: " + str(date.datetime.now()) + "\n")
ml.train(150, 2, "I:/Lab/Real_Life_Event_Logs/BPI_Challenge_2012/data/SplitterByStartingTime/0.2/ReducerRandom/Navarin/inp_referenceLog.csv",
         "I:/Lab/Real_Life_Event_Logs/BPI_Challenge_2012/data/SplitterByStartingTime/0.2/ReducerRandom/Navarin/inp_log_train_red_10.csv",
         "I:/Lab/Real_Life_Event_Logs/BPI_Challenge_2012/data/SplitterByStartingTime/0.2/ReducerRandom/Navarin/inp_log_test.csv",
         "data/BPIC_12/BPIC_12_SplitterTime_02_Random.txt",
         "BPIC_12_SplitterTime_02_Random_10")
with open("data/BPIC_12/BPIC_12_SplitterTime_02_Random.txt", "a") as resultFile:
    resultFile.write("End: " + str(date.datetime.now()) + "\n")