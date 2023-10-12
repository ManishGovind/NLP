import numpy as np

def train_nb(train_x, train_y, vocab):
    loglikelihood = {}
    logprior = 0

    V = len(vocab)
    
    # calculate N_pos and N_neg
    N_pos = N_neg = 0
    for i in range(0,len(train_y)):
        # if the label is positive (greater than zero)
        if train_y[i] == "pos":
            s_p =0
            for k,v in train_x[i].items():
                s_p += v
            N_pos += s_p

        # else, the label is negative
        else:

            s_n =0
            for k,v in train_x[i].items():
                s_n += v
            N_neg += s_n

    
    D = len(train_y)

    
    D_pos = (len(list(filter(lambda x: x == "pos", train_y))))

    # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
    D_neg = D - D_pos

    # Calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)

    # For each word in the vocabulary...
    freq_pos = {}
    freq_neg = {}
    for i in range(0,len(train_y)):
        if train_y[i] == "pos":
            for k,l in train_x[i].items():
                if k in freq_pos :
                    freq_pos[k] += l
                else :
                    freq_pos[k] = 1    
        else :
            for (k,l) in (train_x[i]).items():
                if k in freq_neg :
                    freq_neg[k] += l
                else :
                    freq_neg[k] = 1              


    
    for word , fre in vocab.items():
        
        #print(word)
        f_pos = freq_pos.get(word,0)
        f_neg = freq_neg.get(word,0)
        p_w_pos = (f_pos + 1) / (N_pos + V)
        p_w_neg = (f_neg + 1) / (N_neg + V)
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)

    return logprior, loglikelihood

def nb_predict(doc, logprior, loglikelihood):

    p = 0
    p += logprior

    for word , k in doc.items():
        if word in loglikelihood:
            
            p += loglikelihood[word]

    return p

def test_nb(test_x, test_y, logprior, loglikelihood):
   
    accuracy = 0  
    y_hats = []
    for doc in test_x:
        if nb_predict(doc, logprior, loglikelihood) > 0:
            y_hat_i = "pos"
        else:
            y_hat_i = "neg"
        y_hats.append(y_hat_i)

    cnt  = 0
    for i in range(0,len(test_y)):
        if (test_y[i] == y_hats[i] ):
            cnt+=1
    accuracy = (cnt / len(test_y))
    return accuracy 