# -*- coding: UTF-8 -*-
import random
import jieba
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,hamming_loss,pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import scale
from sklearn.cluster import MiniBatchKMeans
# from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import codecs
import json
import pickle
import logging
from construct import Model
# from tree import HierarchicalClusterAL
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(funcName)s - %(message)s')

def create_data(file,size):
    logging.info("Creating Dataset...")
    cnt = 0
    mapping = { u'书包':0, u'T恤':1, u'阔腿裤':2, u'运动鞋':3}
    examples = []
    target = np.zeros((size,1),dtype=np.int32)
    for line in codecs.open(file,'rb',encoding='utf8'):
        # print(line)
        item = json.loads(line)
        examples.append(item['comment'])
        target[cnt] = np.array(mapping[item['label']])
        cnt += 1

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(examples)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)  
    X_vectors = X_train_tfidf.toarray()

    logging.info("Dataset created\n")
    return X_vectors,target

def leyan_data(file,choice="tfidf"):
    #input format is like text intent-X purpose
    datas_train = []
    labels_train = []
    datas_test = []
    labels_test = []
    train_dis = {}
    test_dis = {}

    for line in codecs.open(file,'rb',encoding='utf8'):
        args = line.split()
        logging.debug(args)
        if args[-1] == "train":
            category = args[-2].split("-")[1]
            if int(category) not in train_dis.keys():
                train_dis[int(category)] = 1
            else:
                train_dis[int(category)] += 1
            text = ""
            for i in range(len(args)-2):
                words = list(jieba.cut(args[i]))
                for word in words:
                    text += (" " + word)
            datas_train.append(text)
            labels_train.append([int(args[-2].split("-")[1])])
        elif args[-1] == "test":
            category = args[-2].split("-")[1]
            if int(category) not in test_dis.keys():
                test_dis[int(category)] = 1
            else:
                test_dis[int(category)] += 1

            text = ""
            for i in range(len(args)-2):
                words = list(jieba.cut(args[i]))
                for word in words:
                    text += (" " + word)
            datas_test.append(text)
            labels_test.append([int(args[-2].split("-")[1])])

    # train_dis = sorted(train_dis.items(),key=lambda x:int(x[0]))
    # test_dis = sorted(test_dis.items(),key=lambda x:int(x[0]))
    # ratio = []
    # for i in range(len(train_dis)):
    #     ratio.append(float(train_dis[i][1])/test_dis[i][1])
    # print(len(train_dis),train_dis)
    # print(len(test_dis),test_dis)
    train_res = np.array(list(train_dis.values()))
    test_res = np.array(list(test_dis.values()))
    train_res = np.true_divide(train_res,np.sum(train_res))
    test_res = np.true_divide(test_res,np.sum(test_res))
    # plt.bar(train_dis.keys(),train_res)
    plt.title("Distribution of Train Data")
    plt.xlabel("Class Type")
    plt.ylabel("Ratio")
    plt.bar(train_dis.keys(),train_res)
    plt.show()
    print(ratio)

    if choice == "word2vec":
        model = Word2Vec.load("leyan.model")
        X_train = []
        X_test = []
        word_vectors = model.wv
        for sentence in datas_train:
            length = 0
            vector = []
            words = sentence.split()
            for word in words:
                if word in word_vectors.vocab:
                    length += 1
                    vector.append(word_vectors[word])
            vector = np.array(vector)
            # logging.debug(vector.shape)
            if length != 0:
                vector = np.array([np.true_divide( np.sum(vector,axis=0), length )])
            else:
                vector = np.zeros(shape=(300,))
            X_train.append(vector)
        for sentence in datas_test:
            length = 0
            words = sentence.split()
            vector = []
            for word in words:
                if word in word_vectors.vocab:
                    length += 1
                    vector.append(word_vectors[word])
            vector = np.array(vector)
            length = vector.shape[0]
            if length != 0:
                vector = np.array([np.true_divide( np.sum(vector,axis=0), length )])
            else:
                vector = np.zeros(shape=(300,))
            X_test.append(vector)

        X_train = np.vstack(X_train)
        # print(X_train)
        X_test = np.vstack(X_test)

        logging.debug(X_train.shape)
        logging.debug(X_test.shape)
        Y_train = np.array(labels_train)
        Y_test = np.array(labels_test)
    else:
        count_vect = CountVectorizer()
        X_counts = count_vect.fit_transform(datas_train+datas_test)
        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X_counts)  
        X_vectors = X_tfidf.toarray()

        logging.debug(X_vectors)
        X_train = X_vectors[:len(datas_train)]
        X_test = X_vectors[len(datas_train):]
        Y_train = np.array(labels_train)
        Y_test = np.array(labels_test)

        logging.debug("X_train: ", str(X_train.shape))
        logging.debug("X_test: ", str(X_test.shape))
    return X_train,Y_train,X_test,Y_test

def mapping(label):
    labels = [[list(i).index(1)] for i in label]
    labels = np.array(labels)
    return labels

def get_result(y_true,y_pred):
    # print("y_true: ",y_true.shape)
    classes = y_true.shape[1]
    # print(y_true,y_pred)
    total = y_true.shape[0]
    # print(total)
    total_correctly_predicted = len([i for i in range(len(y_true)) if (y_true[i]==y_pred[i]).sum() == classes])
    accuracy = float(total_correctly_predicted)/ total
    loss = hamming_loss(y_true,y_pred)

    logging.info("Acurracy %.4f",accuracy)
    logging.info("hamming loss %.4f\n",loss)
    return accuracy,loss

def entropy_learner(x_train , x_test, y_train, y_test,initial_batch,step,model):
    model = Model(model)
    accuracies = []
    losses = []
    l = [initial_batch]

    batch_x, batch_y = x_train[0:initial_batch],y_train[0:initial_batch]
    data,label = x_train[initial_batch:],y_train[initial_batch:]
    # print("batch_x",batch_x.shape)
    # print("data",data.shape)
    model.fit(batch_x, batch_y)
    y_pred = model.predict(x_test)
    # logging.info("*"*100)
    logging.info("Trained on dataset:"+str(batch_x.shape))

    accuracy = accuracy_score(y_test,y_pred)
    loss = hamming_loss(y_test,y_pred)
    logging.info("Acurracy %.4f",accuracy)
    logging.info("hamming loss %.4f\n",loss)

    accuracies.append(accuracy)
    losses.append(loss)


    while data.shape[0]:
        # print("*"*100)
        scores = np.abs(model.predict_proba(data))
        # print("data",data.shape)
        estimates = np.zeros(shape=(data.shape[0],2))

        for i,vector in enumerate(scores):
            entropy = 0.0
            for r in vector:
                # print(r)
                if r!=0:
                    entropy -= abs(r) * np.log(abs(r))
            # print(i,Sum)
            estimates[i] = [int(i),entropy]


        estimates = estimates[estimates[:,1].argsort()][::-1]
        # print("estimates",len(estimates))
        # print(estimates)
        if len(estimates) > step:
            best = estimates[:step,0]
            # print(best)
            # print(type(best[0]))
            best = [int(i) for i in best]
        else:
            best = range(data.shape[0])

        # print(label.shape)
        batch_x = np.vstack((batch_x,data[best]))
        batch_y = np.vstack((batch_y,label[best]))
        # print(batch_y.shape)
        # print(batch_x.shape)
        logging.info("New Dataset shape"+str(batch_x.shape))
        l.append(batch_x.shape[0])

        data = np.delete(data,best,0)
        label = np.delete(label,best,0)
        # print(type(data.shape),len(data.shape))

        model.fit(batch_x,batch_y)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        loss = hamming_loss(y_test,y_pred)
        logging.info("Acurracy %.4f",accuracy)
        logging.info("hamming loss %.4f\n",loss)
        accuracies.append(accuracy)
        losses.append(loss)

    return l,accuracies,losses

def active_learner(x_train , x_test, y_train, y_test,initial_batch,step,model):
    model = Model(model)
    accuracies = []
    losses = []
    l = [initial_batch]

    batch_x, batch_y = x_train[0:initial_batch],y_train[0:initial_batch]
    data,label = x_train[initial_batch:],y_train[initial_batch:]
    # print("data",data.shape)
    # print("label",label.shape)
    model.fit(batch_x, batch_y)
    y_pred = model.predict(x_test)
    # print("*"*100)
    logging.info("Trained on dataset:"+str(batch_x.shape))

    accuracy = accuracy_score(y_test,y_pred)
    logging.info("Acurracy %.4f\n",accuracy)
    accuracies.append(accuracy)


    while data.shape[0]:
        # print("*"*100)
        scores = np.abs(model.predict_proba(data))
        # print("data",data.shape)
        estimates = np.zeros(shape=(data.shape[0],2))

        for i,vector in enumerate(scores):
            # Sum = np.sum(vector)
            Sum = np.max(vector)
            # print(i,Sum)
            estimates[i] = [int(i),Sum]


        estimates = estimates[estimates[:,1].argsort()]
        # print("estimates",len(estimates))
        # print(estimates)
        if len(estimates) > step:
            best = estimates[:step,0]
            # print(best)
            # print(type(best[0]))
            best = [int(i) for i in best]
        else:
            best = range(data.shape[0])

        logging.debug(best)
        logging.debug(label[best].shape)
        logging.debug(batch_y.shape)
        batch_x = np.vstack((batch_x,data[best]))
        batch_y = np.vstack((batch_y,label[best]))
        # print(batch_y.shape)
        # print(batch_x.shape)
        logging.info("New Dataset shape"+str(batch_x.shape))
        logging.debug("New Label shape"+str(batch_y.shape))
        l.append(batch_x.shape[0])

        data = np.delete(data,best,0)
        label = np.delete(label,best,0)
        # print(type(data.shape),len(data.shape))

        logging.info("Fitting")
        model.fit(batch_x,batch_y)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        logging.info("Acurracy %.4f\n",accuracy)
        accuracies.append(accuracy)

    return l,accuracies,losses

def random_learner(x_train , x_test, y_train, y_test,initial_batch,step,model):
    model = Model(model)
    accuracies = []
    losses = []
    l = [initial_batch]

    batch_x, batch_y = x_train[0:initial_batch],y_train[0:initial_batch]
    data,label = x_train[initial_batch:],y_train[initial_batch:]
    # print("fitting")
    model.fit(batch_x, batch_y)
    # print("predicting")
    y_pred = model.predict(x_test)
    # print("*"*100)
    logging.info("Trained on dataset:"+str(batch_x.shape))

    accuracy = accuracy_score(y_test,y_pred)
    loss = hamming_loss(y_test,y_pred)
    logging.info("Acurracy %.4f",accuracy)
    logging.info("hamming loss %.4f\n",loss)
    accuracies.append(accuracy)
    losses.append(loss)


    while data.shape[0]:
        # print("*"*100)

        if data.shape[0] > step:
            index = random.sample(range(data.shape[0]),step)
        else:
            index = range(data.shape[0])
        
        batch_x = np.vstack((batch_x,data[index]))
        batch_y = np.vstack((batch_y,label[index]))
        logging.info("New Dataset shape"+str(batch_x.shape))
        l.append(batch_x.shape[0])

        data = np.delete(data,index,0)
        label = np.delete(label,index,0)

        model.fit(batch_x,batch_y)
        y_pred = model.predict(x_test)
        logging.debug(y_test)
        logging.debug(y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        loss = hamming_loss(y_test,y_pred)
        logging.info("Acurracy %.4f",accuracy)
        logging.info("hamming loss %.4f\n",loss)
        accuracies.append(accuracy)
        losses.append(loss)

    return l,accuracies,losses

def margin_learner(x_train , x_test, y_train, y_test,initial_batch,step,model):
    model = Model(model)
    accuracies = []
    losses = []
    l = [initial_batch]

    batch_x, batch_y = x_train[0:initial_batch],y_train[0:initial_batch]
    data,label = x_train[initial_batch:],y_train[initial_batch:]
    # print("data",data.shape)
    # print("label",label.shape)
    model.fit(batch_x, batch_y)
    y_pred = model.predict(x_test)
    # print("*"*100)
    logging.info("Trained on dataset:"+str(batch_x.shape))

    accuracy = accuracy_score(y_test,y_pred)
    loss = hamming_loss(y_test,y_pred)
    logging.info("Acurracy %.4f",accuracy)
    logging.info("hamming loss %.4f\n",loss)
    accuracies.append(accuracy)
    losses.append(loss)


    while data.shape[0]:
        # print("*"*100)
        scores = np.abs(model.predict_proba(data))
        # scores = model.decision_function(data)
        # print("data",data.shape)
        estimates = np.zeros(shape=(data.shape[0],2))

        for i,vector in enumerate(scores):
            # print(vector)
            tmp = np.argsort(vector)
            # print(vector[tmp[-1]],vector[tmp[-2]])
            margin = vector[tmp[-1]] - vector[tmp[-2]]
            # print(i,Sum)
            estimates[i] = [int(i),margin]


        estimates = estimates[estimates[:,1].argsort()]
        # print("estimates",estimates)
        if len(estimates) > step:
            best = estimates[:step,0]
            # print(best)
            # print(type(best[0]))
            best = [int(i) for i in best]
        else:
            best = range(data.shape[0])

        # print(type(best),type(best[0]))
        batch_x = np.vstack((batch_x,data[best]))
        batch_y = np.vstack((batch_y,label[best]))
        # print(batch_y.shape)
        # print(batch_x.shape)
        logging.info("New Dataset shape"+str(batch_x.shape))
        l.append(batch_x.shape[0])

        data = np.delete(data,best,0)
        # print("after deletion",data.shape)
        label = np.delete(label,best,0)
        # print(type(data.shape),len(data.shape))

        model.fit(batch_x,batch_y)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        loss = hamming_loss(y_test,y_pred)
        logging.info("Acurracy %.4f",accuracy)
        logging.info("hamming loss %.4f\n",loss)
        accuracies.append(accuracy)
        losses.append(loss)

    return l,accuracies,losses

# def diverse_learner(x_train , x_test, y_train, y_test,initial_batch,step,model):

#     if model == "linear":
#         model = LinearSVC()
#     else:
#         model = OneVsRestClassifier(SVC(kernel='rbf',gamma=0.0020,C=5., probability=True ))
#     accuracies = []
#     losses = []
#     l = [initial_batch]

#     batch_x, batch_y = x_train[0:initial_batch],y_train[0:initial_batch]
#     data,label = x_train[initial_batch:],y_train[initial_batch:]
#     # print("data",data.shape)
#     # print("label",label.shape)
#     model.fit(batch_x, batch_y)
#     y_pred = model.predict(x_test)
#     # print("*"*100)
#     logging.info("Trained on dataset:"+str(batch_x.shape))

#     n_clusters = len(np.unique(y_train.tolist()))
#     logging.debug("cluster: ",n_clusters)

#     accuracy = accuracy_score(y_test,y_pred)
#     loss = hamming_loss(y_test,y_pred)
#     logging.info("Acurracy %.4f",accuracy)
#     logging.info("hamming loss %.4f\n",loss)
#     accuracies.append(accuracy)
#     losses.append(loss)

#     while data.shape[0]:

#         logging.info("Constructing clustering model")

#         cluster_model = MiniBatchKMeans(n_clusters= n_clusters)
#         cluster_model.fit(data)
#         unique,counts = np.unique(cluster_model.labels_,return_counts=True)
#         cluster_prob = counts/sum(counts)
#         cluster_labels = cluster_model.labels_
#         # print(len(cluster_labels),cluster_labels)

#         # print("*"*100)
#         scores = np.abs(model.decision_function(data))
#         # scores = model.decision_function(data)
#         # print("data",data.shape)
#         estimates = np.zeros(shape=(data.shape[0],2))

#         for i,vector in enumerate(scores):
#             # print(vector)
#             tmp = np.argsort(vector)
#             # print(vector[tmp[-1]],vector[tmp[-2]])
#             margin = vector[tmp[-1]] - vector[tmp[-2]]
#             # print(i,Sum)
#             estimates[i] = [int(i),margin]


#         estimates = estimates[estimates[:,1].argsort()]
#         best = []
#         best_cnts = [0 for _ in range(n_clusters)]
#         # print("estimates",estimates)
#         if len(estimates) > step:
#             for i,score in estimates:
#                 if len(best) == step:
#                     break
#                 i = int(i)
#                 tmp_label = cluster_labels[i]
#                 logging.debug(tmp_label)
#                 if best_cnts[tmp_label] / step < cluster_prob[tmp_label]:
#                     best.append(i)
#                     best_cnts[tmp_label] += 1
#             n_slot_remaining = step - len(best)
#             batch_filler = list(set(estimates[0]) - set(best))
#             best.extend(batch_filler[0:n_slot_remaining])
#             best = [int(i) for i in best]
#         else:
#             best = range(data.shape[0])

#         # print(type(best),type(best[0]))
#         batch_x = np.vstack((batch_x,data[best]))
#         batch_y = np.vstack((batch_y,label[best]))
#         # print(batch_y.shape)
#         # print(batch_x.shape)
#         logging.info("New Dataset shape"+str(batch_x.shape))
#         l.append(batch_x.shape[0])

#         data = np.delete(data,best,0)
#         # print("after deletion",data.shape)
#         label = np.delete(label,best,0)
#         # print(type(data.shape),len(data.shape))

#         model.fit(batch_x,batch_y)
#         y_pred = model.predict(x_test)
#         accuracy = accuracy_score(y_test,y_pred)
#         loss = hamming_loss(y_test,y_pred)
#         logging.info("Acurracy %.4f",accuracy)
#         logging.info("hamming loss %.4f\n",loss)
#         accuracies.append(accuracy)
#         losses.append(loss)

#     return l,accuracies,losses

def diverse_learner(x_train , x_test, y_train, y_test,initial_batch,step,model):
    model = Model(model)
    accuracies = []
    losses = []
    l = [initial_batch]

    batch_x, batch_y = x_train[0:initial_batch],y_train[0:initial_batch]
    data,label = x_train[initial_batch:],y_train[initial_batch:]
    # print("data",data.shape)
    # print("label",label.shape)
    model.fit(batch_x, batch_y)
    y_pred = model.predict(x_test)
    # print("*"*100)
    logging.info("Trained on dataset:"+str(batch_x.shape))

    n_clusters = len(np.unique(y_train.tolist()))
    logging.debug("cluster: " +str(n_clusters))

    accuracy = accuracy_score(y_test,y_pred)
    loss = hamming_loss(y_test,y_pred)
    logging.info("Acurracy %.4f",accuracy)
    logging.info("hamming loss %.4f\n",loss)
    accuracies.append(accuracy)
    losses.append(loss)

    cluster_model = MiniBatchKMeans(n_clusters= n_clusters)
    cluster_model.fit(data)
    unique,counts = np.unique(cluster_model.labels_,return_counts=True)
    logging.debug(unique)
    logging.debug(counts)
    cluster_prob = counts/np.sum(counts)
    logging.debug(len(counts))
    logging.debug(len(cluster_prob))
    cluster_labels = cluster_model.labels_

    while data.shape[0]:
        logging.debug(data.shape[0])
        logging.info("Constructing clustering model")


        if data.shape[0] > step:
            cnt = {}
            for i in range(len(unique)):
                cnt[i] = 0

            upper = []
            for i in cluster_prob:
                upper.append(i * data.shape[0])

            logging.debug(len(upper))
            logging.debug(len(cnt))

            best = []
            while len(best) < step:
                index = random.choice(range(data.shape[0]))
                lbl = cluster_labels[index]
                logging.debug(str(index)+" "+str(lbl))
                if cnt[lbl] > upper[lbl]:
                    continue
                else:
                    best.append(index)
            best = np.array(best)
        else:
            best = range(data.shape[0])

        logging.debug(best)
        batch_x = np.vstack((batch_x,data[best]))
        batch_y = np.vstack((batch_y,label[best]))
        # print(batch_y.shape)
        # print(batch_x.shape)
        logging.info("New Dataset shape"+str(batch_x.shape))
        l.append(batch_x.shape[0])

        data = np.delete(data,best,0)
        # print("after deletion",data.shape)
        label = np.delete(label,best,0)
        # print(type(data.shape),len(data.shape))

        model.fit(batch_x,batch_y)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        loss = hamming_loss(y_test,y_pred)
        logging.info("Acurracy %.4f",accuracy)
        logging.info("hamming loss %.4f\n",loss)
        accuracies.append(accuracy)
        losses.append(loss)

    return l,accuracies,losses

def center_learner(x_train , x_test, y_train, y_test,initial_batch,step,model):
    model = Model(model)
    accuracies = []
    losses = []
    l = [initial_batch]

    batch_x, batch_y = x_train[0:initial_batch],y_train[0:initial_batch]
    data,label = x_train[initial_batch:],y_train[initial_batch:]
    # print("fitting")
    model.fit(batch_x, batch_y)
    # print("predicting")
    y_pred = model.predict(x_test)
    # print("*"*100)
    logging.info("Trained on dataset:"+str(batch_x.shape))

    accuracy = accuracy_score(y_test,y_pred)
    loss = hamming_loss(y_test,y_pred)
    logging.info("Acurracy %.4f",accuracy)
    logging.info("hamming loss %.4f\n",loss)
    accuracies.append(accuracy)
    losses.append(loss)


    while data.shape[0]:

        if data.shape[0] > step:
            distances = np.zeros(shape=(data.shape[0],2))
            dist = pairwise_distances(data,batch_x,metric="euclidean")
            score = np.min(dist, axis=1)
            for i,num in enumerate(score):
                distances[i] = [int(i),num]
            distances = distances[distances[:,1].argsort()][::-1]
            logging.debug(distances)
            best = distances[:step,0]
            best = [int(i) for i in best]
        else:
            logging.debug("haven't enter kcenter sampling")
            best = range(data.shape[0])

        logging.debug(best)
        batch_x = np.vstack((batch_x,data[best]))
        batch_y = np.vstack((batch_y,label[best]))
        # print(batch_y.shape)
        # print(batch_x.shape)
        logging.info("New Dataset shape"+str(batch_x.shape))
        logging.debug("New Label shape"+str(batch_y.shape))
        l.append(batch_x.shape[0])

        data = np.delete(data,best,0)
        label = np.delete(label,best,0)
        score = np.delete(score,best,0)
        # print(type(data.shape),len(data.shape))

        logging.info("Fitting")
        model.fit(batch_x,batch_y)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        loss = hamming_loss(y_test,y_pred)
        logging.info("Acurracy %.4f",accuracy)
        logging.info("hamming loss %.4f\n",loss)
        accuracies.append(accuracy)
        losses.append(loss)

    return l,accuracies,losses

def hierarchical_learner(x_train , x_test, y_train, y_test,initial_batch,step,model):
    model = Model(model)
    accuracies = []
    losses = []
    l = [initial_batch]

    shape = 0
    batch_x, batch_y = x_train[0:initial_batch],y_train[0:initial_batch]
    labeled = {}
    i = 0
    for row in y_train:
        if i < initial_batch:
            labeled[i] = row[0]
            i += 1
        else:
            break

    model.fit(batch_x,batch_y)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    loss = hamming_loss(y_test,y_pred)
    logging.info("Acurracy %.4f",accuracy)
    logging.info("hamming loss %.4f\n",loss)
    accuracies.append(accuracy)
    losses.append(loss)

    # print(x_t+rain)
    learner = HierarchicalClusterAL(x_train,y_train,0)

    while shape < x_train.shape[0]:
        if x_train.shape[0] - shape < step:
            selected = learner.select_batch_(x_train.shape[0] - shape,batch_x,labeled,y_train)
        else:
            selected = learner.select_batch_(step,batch_x,labeled,y_train)
        batch_x = np.vstack((batch_x,x_train[selected]))
        batch_y = np.vstack((batch_y,y_train[selected]))
        shape = batch_x.shape[0]
        for select in selected:
            labeled[select] = y_train[select][0]

        logging.info("New Dataset shape"+str(batch_x.shape))
        l.append(batch_x.shape[0])

        logging.info("Fitting")
        model.fit(batch_x,batch_y)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        loss = hamming_loss(y_test,y_pred)
        logging.info("Acurracy %.4f",accuracy)
        logging.info("hamming loss %.4f\n",loss)
        accuracies.append(accuracy)
        losses.append(loss)

    return l,accuracies,losses

def show_distribution(y_train,y_test):
    train_dis = {}
    test_dis = {}

    for i in y_test:
        if i[0] not in test_dis.keys():
            test_dis[i[0]] = 1
        else:
            test_dis[i[0]] += 1

    train_res = np.array(list(train_dis.values()))
    test_res = np.array(list(test_dis.values()))
    train_res = np.true_divide(train_res,np.sum(train_res))
    test_res = np.true_divide(test_res,np.sum(test_res))
    # plt.bar(train_dis.keys(),train_res)
    plt.title("Distribution of Test Data")
    plt.xlabel("Class Type")
    plt.ylabel("Ratio")
    plt.bar(test_dis.keys(),test_res,width=0.5,color='lightskyblue')
    plt.show()
    print(ratio)

def main(initial_batch,step,model,methods,input_choice,plot):

    # f = h5py.File("dataset_294.h5")
    # x = f['x'].value
    # y = f['y'].value
    # f.close()
    # x = scale(x)
    # y = mapping(y)

    logging.info("On the dataset of Taobao comments")
    x,y = create_data('items.json',10872)
    # x,y = create_data('toy.json',1000)
    # x,y = create_data('try.json',80)
    # x = pd.DataFrame(x)
    # y = pd.DataFrame(y)
    x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
    show_distribution(y_train,y_test)
    # x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
    # logging.debug(x_train)
    # logging.info("On the dataset of leyan")
    # x_train , y_train, x_test, y_test = leyan_data("yanjing.anonymous.replace.txt",input_choice)
    # logging.info("Begin training")

    if methods == "all":
        l1,accuracy1,loss1 = active_learner(x_train , x_test, y_train, y_test,initial_batch,step,model)
        l2,accuracy2,loss2 = random_learner(x_train , x_test, y_train, y_test,initial_batch,step,model)
        l3,accuracy3,loss3 = entropy_learner(x_train , x_test, y_train, y_test,initial_batch,step,model)
        l4,accuracy4,loss4 = margin_learner(x_train , x_test, y_train, y_test,initial_batch,step,model)
        # l5,accuracy5,loss5 = hierarchical_learner(x_train , x_test, y_train, y_test,initial_batch,step,model)
        l6,accuracy6,loss6 = center_learner(x_train , x_test, y_train, y_test,initial_batch,step,model)
        if plot:
            plt.plot(l1,accuracy1,label = "active",color="red")
            plt.plot(l2,accuracy2,label = "random",color="black")
            plt.plot(l3,accuracy3,label = "entropy",color="blue")
            plt.plot(l4,accuracy4,label = "margin",color="green")
            # plt.plot(l5,accuracy5,label = "hierarchical",color="yellow")
            plt.plot(l6,accuracy6,label = "kcenter",color="purple")
            plt.legend()
            plt.show()
        else:
            with open("active_"+model+".txt","w") as f:
                for i in l1:
                    f.write(str(i)+'\t')
                f.write("\n")
                for i in accuracy1:
                    f.write(str(i)+'\t')
                f.write("\n")  
                for i in loss1:
                    f.write(str(i)+'\t')
                f.write("\n") 
            with open('random_'+model+'.txt','w') as f:
                for i in l2:
                    f.write(str(i)+'\t')
                f.write("\n")
                for i in accuracy2:
                    f.write(str(i)+'\t')
                f.write("\n")  
                for i in loss2:
                    f.write(str(i)+'\t')
                f.write("\n")
            with open('entropy_'+model+'.txt','w') as f:
                for i in l3:
                    f.write(str(i)+'\t')
                f.write("\n")
                for i in accuracy3:
                    f.write(str(i)+'\t')
                f.write("\n")  
                for i in loss3:
                    f.write(str(i)+'\t')
                f.write("\n") 
            with open('margin_'+model+'.txt','w') as f:
                for i in l4:
                    f.write(str(i)+'\t')
                f.write("\n")
                for i in accuracy4:
                    f.write(str(i)+'\t')
                f.write("\n")  
                for i in loss4:
                    f.write(str(i)+'\t')
                f.write("\n") 
            with open('center_'+model+'.txt','w') as f:
                for i in l6:
                    f.write(str(i)+'\t')
                f.write("\n")
                for i in accuracy6:
                    f.write(str(i)+'\t')
                f.write("\n")  
                for i in loss6:
                    f.write(str(i)+'\t')
                f.write("\n") 
    else:
        function = methods+"_learner"
        file = methods + ".txt"
        l,accuracy,loss = globals()[function](x_train , x_test, y_train, y_test,initial_batch,step,model)
        if plot:
            plt.plot(l,accuracy,l,loss,label = file,color="red")
            plt.show()
        else:
            logging.debug(file)
            with open(file,'w') as f:
                for i in l:
                    f.write(str(i)+'\t')
                f.write("\n")
                for i in accuracy:
                    f.write(str(i)+'\t')
                f.write("\n")  
                for i in loss:
                    f.write(str(i)+'\t')
                f.write("\n")           

def plot(files):
    plt.title('Approaches on Logistic Regression')
    colors = ["#ef5350","#595959","#2979ff","gray","#ffa000","purple"]
    i = 0
    for file in files:
        cnt = 0
        paras = [[],[],[]]
        with open(file,'r') as f:
            for line in f.readlines():
                datas = line.split()
                c = 0
                for data in datas:
                    # if c > 97:
                        # break
                    paras[cnt].append(float(data))
                    c += 1
                # print(paras[cnt])
                cnt += 1
        plt.plot(paras[0],paras[1],label = file.split(".")[0].split('/')[-1].split('_')[0] ,color=colors[i])
        i += 1
    plt.xlabel('Labelled')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(linestyle='--')
    plt.show()

if __name__ == '__main__':

    # main(5000,1000,'linear','hierarchical',False)
    # main(500,100,'linear','hierarchical',False)
    # main(300,100,'linear','all', "word2vec",  False)
    # main(300,100,'logistic','all', "tfidf",  False)
    # main(5,10,'linear','all',True)

    # plot(["data/taobao/active_logistic.txt","data/taobao/random_logistic.txt","data/taobao/entropy_logistic.txt","data/taobao/margin_logistic.txt","data/taobao/center_logistic.txt"])
    # plot(["data/taobao/active_linear.txt","data/taobao/random_linear.txt","data/taobao/entropy_linear.txt","data/taobao/margin_linear.txt","data/taobao/center_linear.txt"])
    plot(["data/taobao/active_bayes.txt","data/taobao/random_bayes.txt","data/taobao/entropy_bayes.txt","data/taobao/margin_bayes.txt","data/taobao/center_bayes.txt"])

    # leyan_data("yanjing.anonymous.replace.txt")