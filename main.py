import os
import random
import math 

######## HELPER FUNCTIONS ##########################
def Kfold(data, k):
    k_folds = []
    length = int(len(data)/k) 
    for i in range(k):
        k_folds += [data[i*length:(i+1)*length]] 
    return k_folds
def prepare_vocab(text):
    vocab = {}
    words = list(set(text.split()))

    for i, word in enumerate(words):
        vocab[word] = i

    return vocab   
def tokenise_fold_counts(fold,vocab):
    counts = dict()
    words = fold.split()

    for word in words:
        if vocab[word] in counts:
            counts[vocab[word]] += 1
        else:
            counts[vocab[word]] = 1

    return counts    
    
######################################################    
############# READING FILES ##########################
pos_features = []                                           #positive Features
neg_features = []                                           #negative features

files_positive = os.listdir('txt_sentoken/pos/')
files_negative = os.listdir('txt_sentoken/neg/')

for pos in files_positive:
    with open('txt_sentoken/pos/' + pos) as f:
        pos_features.append(f.read())

for neg in files_negative:
    with open('txt_sentoken/neg/' + neg) as f:
        neg_features.append(f.read())     

########################################################     
# 
# ########## Kfold #################################


    
X = pos_features + neg_features
y = [1] * len(pos_features) + [0] * len(neg_features)

temp = list(zip(X, y))                                       #shuffling the data before making the folds
random.shuffle(temp)
X, y = zip(*temp)
folds = 10



X_fold = Kfold(X, folds)
y_fold = Kfold(y, folds)
accuracy = []
precision = []
recalls = []
f1 = []
for k in range(folds):
 ############ Calculating Prior Probability ####################   
    pos = 0
    neg = 0
    for i in range(folds):
        if i != k:
            for j in y_fold[i]:
                if j == 1:
                    pos += 1
                else:
                    neg += 1
    prob_pos = pos/(pos + neg)
    prob_neg = neg/(pos + neg)  
#####################################################################
# ############ tokenising text and getting counts ########################                      

    text_pos = ''
    text_neg = ''
    for i in range(folds):
        if i != k:
            for j in range(len(X_fold[i])):
                if y_fold[i][j] == 1:
                    text_pos += ' ' + X_fold[i][j]
                else:
                    text_neg += ' ' + X_fold[i][j]
    vocab = prepare_vocab(text_pos +' '+  text_neg)

    pos_counts = tokenise_fold_counts(text_pos, vocab)
    neg_counts = tokenise_fold_counts(text_neg, vocab)                    
####################################################################
################ calculating Probabilities of each word #########################
    prob_pos_dict = {}
    pos_total_count = 0
    for i in pos_counts.values():
        pos_total_count += i +1
    for i in pos_counts.keys():
        prob_pos_dict[i] = (pos_counts[i]  +1) /  (pos_total_count )

    prob_neg_dict = {}
    neg_total_count = 0
    for i in neg_counts.values():
        neg_total_count += i+1 
    for i in neg_counts.keys():
        prob_neg_dict[i] = (neg_counts[i] +1)/  (neg_total_count )
#####################################################################
###################### Calculating probabilities of our test data##########################
    y_label = []
    base = 100
    for i in X_fold[k]:
        pos_p = 1
        neg_p = 1
        
        for j in i.split():
            try:
                pos_p *= math.log(prob_pos_dict[vocab[j]] , base)
            except:    
                pos_p *= math.log(1/pos_total_count , base)    

            try:
                neg_p *= math.log(prob_neg_dict[vocab[j]] , base)

            except:              
                neg_p *= math.log(1/neg_total_count , base)    

        if abs(pos_p * math.log(prob_pos , base)) < abs(neg_p * math.log(prob_neg , base)):
            y_label.append(1)
        else:
            y_label.append(0)
 ################################################################################  
 # ######################Calculating metrics ##################################
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_label)):
        if y_label[i] == 0 and y_fold[k][i] == 0:
            tn += 1
        elif y_label[i] == 1 and y_fold[k][i] == 1:
            tp += 1
        elif y_label[i] == 1 and y_fold[k][i] == 0:
            fn += 1
        elif y_label[i] == 0 and y_fold[k][i] == 1:
            fp += 1
    print('Current Fold : {} out of {} folds'.format(k+1,folds))        
    print('Accuracy: {} %'.format(100*(tp+tn)/(tp+tn+fp+fn)))
    print('Precision: {} %'.format(100*(tp)/(tp+fp)))
    print('Recall: {} %'.format(100*(tp)/(tp+fn)))
    accr = (tp+tn)/(tp+tn+fp+fn)
    pre = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    accuracy.append(accr)
    precision.append(pre)
    recalls.append(recall)
    f1.append(2* (pre * recall)/ (pre + recall))
    print('F1 - Score: {} %\n'.format(100*2* (pre * recall)/ (pre + recall)))

print('\nAvg. Accuracy: {} %'.format(100*sum(accuracy)/ len(accuracy)))
print('Avg. Precision: {} %'.format(100*sum(precision)/ len(precision)))
print('Avg. Recall: {} %'.format(100*sum(recalls)/ len(recalls)))
print('Avg. F1 Score: {} %'.format(100*sum(recalls)/ len(recalls)))






       

