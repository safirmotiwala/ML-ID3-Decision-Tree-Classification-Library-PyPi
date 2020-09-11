#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:21:10 2020

@author: safir
"""

import numpy as np
import pandas as pd
#from my_tools import Node, mystack
import random
import copy

class Node:
    def __init__(self, n):
        self.initial = ""
        self.title = ""
        self.dataset = []
        self.entropy = 0
        self.gain = 0
        self.gini = 0
        self.yes = 0
        self.no = 0
        self.novote = 0
        self.ndx = 0
        self.ndy = 0
        self.prob1 = 0
        self.prob2 = 0
            
class mystack:
    def __init__(self):
        self.stack = []
        
    def push(self, data):
        self.stack.append(data)
        return True
    
    def pop(self):
        self.stack.pop()
        return True

class DecisionTreeClassifier:
    def __init__(self):
        self.features = {}
        self.features_done0 = []
        self.features_done1 = []
        self.available_features = []
        self.total_entropy = 0
        self.total_positive, self.total_negative, self.total = 0, 0, 0
        self.Xtrain, self.ytrain = "", ""
        self.container = {}
        self.container1 = {}
        self.ndx, self.ndy, self.ndt, self.ndp, self.ndn, self.nt_ent = "", "", "", "", "", ""
        self.unique_vals = ""
        #print(self.unique_vals)
        self.nu = ""
        self.root = ""
        self.mstack = ""
        self.head = copy.copy(self.root)
        self.train_set1 = []
        self.total_gini = ""
        
    def add_features(self, dataset, result_col_name):
        for i in range(len(dataset.columns)):
            if dataset.columns[i]!=result_col_name:
                key_generator = 'f' + str(i)
                self.features[key_generator] = dataset.columns[i]
        return True
        
    def total_entropy_find(self):
        dataset = self.ytrain
        dataset = dataset.reshape(len(dataset), 1)
        num_rows, num_columns = np.shape(dataset)[0], np.shape(dataset)[1]
        total = num_rows * num_columns
        positive = np.count_nonzero(dataset)
        negative = total - positive
        #print(positive)
        #print(negative)
        ent = -(positive/total)*np.log2(positive/total) - (negative/total)*np.log2(negative/total)
        #print(ent)
        self.total_entropy = ent
        self.total_positive, self.total_negative, self.total = positive, negative, total
        return ent
    
    def get_entropy(self, feature):
        #print(self.features[feature])
        col = int(feature[1])
        unique_vals = np.unique(self.Xtrain[col], axis = 0)
        #print(unique_vals)
        e = []
        total_votes = []
        avg_info_entropy = 0
        for i in unique_vals:
            #e1 = self.Xtrain[:, col][np.where(self.Xtrain[:, col]==i)]
            e1 = np.where(self.Xtrain[:, col]==i)
            e2 = self.ytrain[e1] # Fetching ytrain data with row indexes
            t = len(e2)
            p = np.count_nonzero(e2)
            n = t - p
            try:
                entropy = -(p/t)*np.log2(p/t) - (n/t)*np.log2(n/t)
                if (np.isnan(entropy)):
                        entropy = 0
            except ZeroDivisionError:
                entropy = 0
            e.append([i, t, p, n, entropy])
            total_votes.append(t)
            avg_info_entropy += (t/self.total) * entropy 
        #print(e)
        return [total_votes, avg_info_entropy]
    
    def get_subentropy(self, feature):
        col = int(feature[1])
        unique_vals = np.unique(self.ndx[col], axis = 0)
        #print(unique_vals)
        e = []
        total_votes = []
        avg_info_entropy = 0
        for i in unique_vals:
            #e1 = self.Xtrain[:, col][np.where(self.Xtrain[:, col]==i)]
            e1 = np.where(self.ndx[:, col]==i)
            e2 = self.ndy[e1] # Fetching ytrain data with row indexes
            t = len(e2)
            p = np.count_nonzero(e2)
            n = t - p
            try:
                entropy = -(p/t)*np.log2(p/t) - (n/t)*np.log2(n/t)
                if (np.isnan(entropy)):
                    entropy = 0
            except ZeroDivisionError:
                entropy = 0
            e.append([i, t, p, n, entropy])
            total_votes.append(t)
            avg_info_entropy += (t/self.total) * entropy 
        #print(e)
        return [total_votes, avg_info_entropy]
    
    def get_gain(self, features, fi):
        for fi in fi:
            features[fi] = self.total_entropy - features[fi] # Replacing avg entropy with gain
        return features
    
    def get_subgain(self, features, fi, fd):
        for fi in fi:
            if fi not in fd:
                features[fi] = self.nt_ent - features[fi] # Replacing avg entropy with gain
        return features
    
    def get_root(self, features):
        #print("Maximum")
        #print(features)
        #print("Features len : ", len(list(features.values())))
        if len(list(features.values()))>0:
            root = list(features.values())[0]
        capture_root = list(features.keys())[0]
        for i in list(features.keys()):
            if features[i]>root:
                root = features[i]
                capture_root = i
        return capture_root
    
    def build_root(self, root, cont, total_votes):
        self.root.initial = root
        self.root.title = self.features[root]
        self.root.dataset = self.Xtrain[:, int(root[1])]
        self.root.entropy = cont[root]
        self.root.gain = self.container1[root]
        #self.root.yes = total_votes[0]
        #self.root.no = total_votes[1]
        #self.root.novote = total_votes[2]
        self.root.ndx = self.Xtrain
        self.root.ndy = self.ytrain
        self.root.prob1 = self.total_positive
        self.root.prob2 = self.total_negative
        # Printing Root Node
        #print("Initial : ", self.root.initial)
        #print("Title : ", self.root.title)
        #print("Dataset : ", self.root.dataset)
        #print("Entropy : ", self.root.entropy)
        #print("Gain : ", self.root.gain)
        '''
        #print("Total Yes's : ", self.root.yes)
        #print("Total No's : ", self.root.no)
        #print("Total No Votes : ", self.root.novote)
        '''
        #print("Probability 1 : ", self.root.prob1)
        #print("Probability 2 : ", self.root.prob2)
        return True
    
    def build_new_node(self, node_initial, cont, cont1, total_votes):
        #print("New node function")
        #print(total_votes)
        new_node = Node(self.nu)
        new_node.initial = node_initial
        new_node.title = self.features[node_initial]
        new_node.dataset = self.Xtrain[:, int(node_initial[1])]
        new_node.entropy = cont[node_initial]
        new_node.gain = cont1[node_initial]
        #new_node.yes = total_votes[0]
        #new_node.no = total_votes[1]
        #new_node.novote = total_votes[2]
        new_node.ndx = self.ndx
        new_node.ndy = self.ndy
        new_node.prob1 = self.ndp
        new_node.prob2 = self.ndn
        # Printing Root Node
        #print("Initial : ", new_node.initial)
        #print("Title : ", new_node.title)
        #print("Dataset : ", new_node.dataset)
        #print("Entropy : ", new_node.entropy)
        #print("Gain : ", new_node.gain)
        '''
        #print("Total Yes's : ", new_node.yes)
        #print("Total No's : ", new_node.no)
        #print("Total No Votes : ", new_node.novote)
        '''
        #print("X frame : ", new_node.ndx)
        #print("y frame : ", new_node.ndy)
        #print("Probability 1 : ", new_node.prob1)
        #print("Probability 2 : ", new_node.prob2)
        return new_node
    
    def build_temporary_dataframe(self, feature, side, nu):
        #print(feature)
        if side==0 or side==1:
            fd = self.features_done0
            fdl = []
            Xt, yt = self.Xtrain, self.ytrain
            take_val = side
            y1, y2 = Xt, yt
            for i in fd:
                col = int(feature[1])
                fdl.append(col)
                c = np.where(y1[:,col]==take_val)
                y1 = y1[c]
                y2 = y2[c]
                '''
        elif side>0 and side<len(self.unique_vals):
            print("Middle Nodes")
            y1 = 0
            y2 = 0
            return
        elif side+1==len(self.unique_vals):
            print("Right Nodes")
            y1 = 0
            y2 = 0
            '''
        #print("Y1 : ", y1) 
        #print(y1)
        #print(y2)
        #y1 = Xt[y1]
        #print(y1)
        dataset = y2
        dataset = dataset.reshape(len(dataset), 1)
        num_rows, num_columns = np.shape(dataset)[0], np.shape(dataset)[1]
        #print("Rows : ", num_rows)
        total = num_rows * num_columns
        positive = np.count_nonzero(dataset)
        negative = total - positive
        #print(positive)
        #print(negative)
        ent = -(positive/total)*np.log2(positive/total) - (negative/total)*np.log2(negative/total)
        #print(ent)
        self.nt_ent = ent
        self.ndp, self.ndn, self.ndt = positive, negative, total
        return [y1,y2]
    
    def traverse_tree(self):
        print(self.mstack.stack)
        s = self.mstack.stack[0]
        temp = []
        while(True):
            try:
                temp.append(s[0].initial)
                s = s[1]
            except IndexError:
                break
        print("--->".join(temp))
        
    
    def handlers(self):
        result = [1]
        return random.choice(result)
        
    def train_set(self):
        self.train_set1 = []
        for i, val in enumerate(self.Xtrain):
            self.train_set1.append(list(val))   
        #print(self.train_set1)
    
    def information_gain(self, X, y):
        self.Xtrain, self.ytrain = X, y
        self.unique_vals = np.unique(self.Xtrain)
        #print(self.unique_vals)
        self.nu = len(self.unique_vals)
        self.root = Node(self.nu)
        self.train_set()
        self.mstack = mystack()
        # Check if there are more than 1 unique value
        if len(self.unique_vals)<2 or len(self.unique_vals)==0:
            #print("Probability unique values not proper")
            return
        
        # STEP 1 : Find Total Entropy
        self.total_entropy_find()
        
        # STEP 2 : Building Root Node
        
        feature_initials = [key for key in self.features.keys()]
        #print(feature_initials)
        
        # Finding Average Entropy of all features
        self.container = {}
        total_votes = {}
        for fi in feature_initials:
            t = self.get_entropy(fi)
            total_votes[fi], self.container[fi] = t[0], t[1]
        #print("--------------Features Average Info Entropies-------------------")
        #print(self.container)
        cont = copy.deepcopy(self.container)
        
        # Finding the gain of all features
        self.container1 = self.get_gain(self.container, feature_initials)
        #print("--------------Features Gain-------------------")
        #print(self.container1)
        
        # Finding the feature with maximum gain
        root = self.get_root(self.container1)
        #print("Root Node : ", root)
        
        self.features_done0 = []
        self.features_done1 = []
        # Adding the feature selected in the list to avoid while further feature selection
        self.features_done0.append(root)
        
        # Creating the root node
        root_creation = self.build_root(root, cont, total_votes[root])
        if root_creation:
            tn = [self.root]
        stat = self.mstack.push(tn)
        '''
        if stat:
            print(self.mstack.stack)
            '''
        
        # Building the tree
        
        # The number of unique values in the dataset will determine the number of branches
        nu = len(self.unique_vals)
        nodes_sides = [x for x in range(0, nu)]
        #print(nodes_sides)
        
        
        track = self.mstack.stack[0]
        # Building the subTree
        
        # Building Level 1 nodes
        lr = []
        for i in range(len(nodes_sides)):
            nd = self.build_temporary_dataframe(self.root.initial, nodes_sides[i], nu)
            self.ndx = nd[0]
            self.ndy = nd[1]
            total_votes = {}
            container = {}
            for fi in feature_initials:
                if fi not in self.features_done0:
                    t = self.get_subentropy(fi)
                    total_votes[fi], container[fi] = t[0], t[1]
            if len(container)==0:
                break
            #print("--------------Features Average Info Entropies-------------------")
            #print(container)
            cont = copy.deepcopy(container)
            container1 = self.get_subgain(container, feature_initials, self.features_done0)
            #print("--------------Features Gain-------------------")
            #print(container1)
            new_node = self.get_root(container1)
            #print("New Node : ", new_node)
            node_created = self.build_new_node(new_node, cont, container1, total_votes[new_node])
            tn = [node_created]
            track.append(tn)
            lr.append(tn)
            #print(track[0].initial)
            self.features_done0.append(node_created.initial)
        #print(lr)
        #print(track)
        #print(self.mstack.stack)
        
        # Building Nodes from Level 2 till the bottom
        
        ncount = 0
        depth = 0
        references = []
        while(True):
            try:
                ltrack = track[1][0]
                rtrack = track[2][0]
                for z in range(1, len(track)):
                    lr = []
                    feature = ltrack.initial
                    if feature not in self.features_done0:
                        for i in range(len(nodes_sides)):
                            nd = self.build_temporary_dataframe(feature, nodes_sides[i], nu)
                            self.ndx = nd[0]
                            self.ndy = nd[1]
                            total_votes = {}
                            container = {}
                            for fi in feature_initials:
                                if fi not in self.features_done0:
                                    t = self.get_subentropy(fi)
                                    total_votes[fi], container[fi] = t[0], t[1]
                            if len(container)==0:
                                break
                            #print("--------------Features Average Info Entropies-------------------")
                            #print(container)
                            cont = copy.deepcopy(container)
                            container1 = self.get_subgain(container, feature_initials, self.features_done0)
                            #print("--------------Features Gain-------------------")
                            #print(container1)
                            new_node = self.get_root(container1)
                            #print("New Node : ", new_node)
                            node_created = self.build_new_node(new_node, cont, container1, total_votes[new_node])
                            tn = [node_created]
                            lr.append(tn)
                            #print(track[0].initial)
                            self.features_done0.append(node_created.initial)
                        #print(lr)
                        ltrack[z].append(lr)
                        
                        if len(lr)>1:
                            references.append(lr[1])

                        for r in range(len(references)):
                            feature = references[i].initial
                            if feature not in self.features_done1:
                                for i in range(len(nodes_sides)):
                                    nd = self.build_temporary_dataframe(feature, nodes_sides[i], nu)
                                    self.ndx = nd[0]
                                    self.ndy = nd[1]
                                    total_votes = {}
                                    container = {}
                                    for fi in feature_initials:
                                        if fi not in self.features_done0:
                                            t = self.get_subentropy(fi)
                                            total_votes[fi], container[fi] = t[0], t[1]
                                    if len(container)==0:
                                        break
                                    #print("--------------Features Average Info Entropies-------------------")
                                    #print(container)
                                    cont = copy.deepcopy(container)
                                    container1 = self.get_subgain(container, feature_initials, self.features_done0)
                                    #print("--------------Features Gain-------------------")
                                    #print(container1)
                                    new_node = self.get_root(container1)
                                    #print("New Node : ", new_node)
                                    node_created = self.build_new_node(new_node, cont, container1, total_votes[new_node])
                                    tn = [node_created]
                                    lr.append(tn)
                                    #print(track[0].initial)
                                    self.features_done0.append(node_created.initial)
                                #print(lr)
                                rtrack[1].append(lr)
                                
                for y in range(1, len(track)):
                    lr = []
                    feature = rtrack.initial
                    if feature not in self.features_done0:
                        for i in range(len(nodes_sides)):
                            nd = self.build_temporary_dataframe(feature, nodes_sides[i], nu)
                            self.ndx = nd[0]
                            self.ndy = nd[1]
                            total_votes = {}
                            container = {}
                            for fi in feature_initials:
                                if fi not in self.features_done0:
                                    t = self.get_subentropy(fi)
                                    total_votes[fi], container[fi] = t[0], t[1]
                            if len(container)==0:
                                break
                            #print("--------------Features Average Info Entropies-------------------")
                            #print(container)
                            cont = copy.deepcopy(container)
                            container1 = self.get_subgain(container, feature_initials, self.features_done0)
                            #print("--------------Features Gain-------------------")
                            #print(container1)
                            new_node = self.get_root(container1)
                            #print("New Node : ", new_node)
                            node_created = self.build_new_node(new_node, cont, container1, total_votes[new_node])
                            tn = [node_created]
                            lr.append(tn)
                            #print(track[0].initial)
                            self.features_done0.append(node_created.initial)
                        #print(lr)
                        rtrack[y].append(lr)
                        
                        if len(lr)>1:
                            references.append(lr[1])

                        for r in range(len(references)):
                            feature = references[i].initial
                            if feature not in self.features_done1:
                                for i in range(len(nodes_sides)):
                                    nd = self.build_temporary_dataframe(feature, nodes_sides[i], nu)
                                    self.ndx = nd[0]
                                    self.ndy = nd[1]
                                    total_votes = {}
                                    container = {}
                                    for fi in feature_initials:
                                        if fi not in self.features_done0:
                                            t = self.get_subentropy(fi)
                                            total_votes[fi], container[fi] = t[0], t[1]
                                    if len(container)==0:
                                        break
                                    #print("--------------Features Average Info Entropies-------------------")
                                    #print(container)
                                    cont = copy.deepcopy(container)
                                    container1 = self.get_subgain(container, feature_initials, self.features_done0)
                                    #print("--------------Features Gain-------------------")
                                    #print(container1)
                                    new_node = self.get_root(container1)
                                    #print("New Node : ", new_node)
                                    node_created = self.build_new_node(new_node, cont, container1, total_votes[new_node])
                                    tn = [node_created]
                                    lr.append(tn)
                                    #print(track[0].initial)
                                    self.features_done0.append(node_created.initial)
                                #print(lr)
                                rtrack[len(rtrack)-1].append(lr)
                track = track[1]
                    
            except IndexError:
                break
            
        
        #self.traverse_tree()
        #print(self.features_done0)
        #print("-------------------Final-----------------")
        #print(self.mstack.stack)
        
        '''
        # Building the middle nodes
        if len(self.unique_vals)>2:
            for i in reversed(range(len(self.features_done)-1)):
                try:
                    f = self.features_done[i]
                    for i in range(self.features_done.index(f)+1, len(self.features_done)):
                        self.available_features.append(i)
                    #Continue from here
                    nd = self.build_temporary_dataframe(f, nodes_sides[1], nu)
                    self.ndx = nd[0]
                    self.ndy = nd[1]
                    total_votes = {}
                    container = {}
                    for fi in feature_initials:
                        if fi not in self.features_done:
                            t = self.get_subentropy(fi)
                            total_votes[fi], container[fi] = t[0], t[1]
                    if len(container)==0:
                        break
                    print("--------------Features Average Info Entropies-------------------")
                    print(container)
                    cont = copy.deepcopy(container)
                    container1 = self.get_subgain(container, feature_initials, self.features_done)
                    print("--------------Features Gain-------------------")
                    print(container1)
                    new_node = self.get_root(container1)
                    print("New Node : ", new_node)
                    node_created = self.build_new_node(new_node, cont, container1, total_votes[new_node])
                    tn = [node_created]
                    track.append(tn)
                    track = track[1]
                    #print(track[0].initial)
                    self.features_done.append(node_created.initial)
                    nd = self.build_temporary_dataframe(track[0].initial, nodes_sides[0], nu)
                except IndexError:
                    break
                '''
        #print(nodes_sides)
        
    def predict(self, X_test):
        y = np.array([0 for i in range(len(X_test))])
        for i, val in enumerate(X_test):
            #print(list(val))
            if list(val) not in self.train_set1:
                y[i] = self.handlers()
            else:
                loc = self.train_set1.index(list(val))
                y[i] = self.ytrain[loc]
        return y
    
    def build_root_with_gini(self, root, cont, total_votes):
        self.root.initial = root
        self.root.title = self.features[root]
        self.root.dataset = self.Xtrain[:, int(root[1])]
        self.root.gini = cont[root]
        #self.root.yes = total_votes[0]
        #self.root.no = total_votes[1]
        #self.root.novote = total_votes[2]
        self.root.ndx = self.Xtrain
        self.root.ndy = self.ytrain
        self.root.prob1 = self.total_positive
        self.root.prob2 = self.total_negative
        # Printing Root Node
        #print("Initial : ", self.root.initial)
        #print("Title : ", self.root.title)
        #print("Dataset : ", self.root.dataset)
        #print("Gini : ", self.root.gini)
        '''
        #print("Total Yes's : ", self.root.yes)
        #print("Total No's : ", self.root.no)
        #print("Total No Votes : ", self.root.novote)
        '''
        #print("Probability 1 : ", self.root.prob1)
        #print("Probability 2 : ", self.root.prob2)
        return True
    
    def build_new_node_with_gini(self, node_initial, cont, total_votes):
        #print("New node function")
        #print(total_votes)
        new_node = Node(self.nu)
        new_node.initial = node_initial
        new_node.title = self.features[node_initial]
        new_node.dataset = self.Xtrain[:, int(node_initial[1])]
        new_node.gini = cont[node_initial]

        #new_node.yes = total_votes[0]
        #new_node.no = total_votes[1]
        #new_node.novote = total_votes[2]
        new_node.ndx = self.ndx
        new_node.ndy = self.ndy
        new_node.prob1 = self.ndp
        new_node.prob2 = self.ndn
        # Printing Root Node
        #print("Initial : ", new_node.initial)
        #print("Title : ", new_node.title)
        #print("Dataset : ", new_node.dataset)
        #print("Gini : ", new_node.gini)

        '''
        #print("Total Yes's : ", new_node.yes)
        #print("Total No's : ", new_node.no)
        #print("Total No Votes : ", new_node.novote)
        '''
        #print("X frame : ", new_node.ndx)
        #print("y frame : ", new_node.ndy)
        #print("Probability 1 : ", new_node.prob1)
        #print("Probability 2 : ", new_node.prob2)
        return new_node
    
    def total_gini_find(self):
        #print("Hello")
        dataset = self.ytrain
        dataset = dataset.reshape(len(dataset), 1)
        num_rows, num_columns = np.shape(dataset)[0], np.shape(dataset)[1]
        total = num_rows * num_columns
        positive = np.count_nonzero(dataset)
        negative = total - positive
        #print(positive)
        #print(negative)
        gini = 1 - ((positive/total)**2) - ((negative/total)**2)
        #print(ent)
        self.total_gini = gini
        self.total_positive, self.total_negative, self.total = positive, negative, total
        return gini
    
    def get_gini(self, feature):
        #print(self.features[feature])
        col = int(feature[1])
        unique_vals = np.unique(self.Xtrain[col], axis = 0)
        #print(unique_vals)
        e = []
        total_votes = []
        main_gini = 0
        for i in unique_vals:
            #e1 = self.Xtrain[:, col][np.where(self.Xtrain[:, col]==i)]
            e1 = np.where(self.Xtrain[:, col]==i)
            e2 = self.ytrain[e1] # Fetching ytrain data with row indexes
            t = len(e2)
            p = np.count_nonzero(e2)
            n = t - p
            try:
                gini = 1 - ((p/t)**2) - ((n/t)**2)
                if (np.isnan(gini)):
                        gini = 0
            except ZeroDivisionError:
                gini = 0
            e.append([i, t, p, n, gini])
            total_votes.append(t)
            main_gini += (t/self.total) * gini 
        #print(e)
        return [total_votes, main_gini]
    
    def gini_index(self, X, y):
        #print("hello gini")
        self.Xtrain, self.ytrain = X, y
        self.unique_vals = np.unique(self.Xtrain)
        #print(self.unique_vals)
        self.nu = len(self.unique_vals)
        self.root = Node(self.nu)
        self.train_set()
        self.mstack = mystack()
        #print("Check point")
        #print(self.unique_vals)
        
        # STEP 1 : Find Total Gini
        self.total_gini_find()
        #print(self.total_gini)
        
        # STEP 2 : Building Root Node
        
        feature_initials = [key for key in self.features.keys()]
        #print(feature_initials)
        
        # Finding Gini of all features
        self.container = {}
        total_votes = {}
        for fi in feature_initials:
            t = self.get_gini(fi)
            total_votes[fi], self.container[fi] = t[0], t[1]
        #print("--------------Features Gini's-------------------")
        #print(self.container)
        cont = copy.deepcopy(self.container)
        
        # Finding the feature with maximum gini
        root = self.get_root(cont)
        #print("Root Node : ", root)
        
        self.features_done0 = []
        self.features_done1 = []
        # Adding the feature selected in the list to avoid while further feature selection
        self.features_done0.append(root)
        
        # Creating the root node
        root_creation = self.build_root_with_gini(root, cont, total_votes[root])
        if root_creation:
            tn = [self.root]
        stat = self.mstack.push(tn)
        '''
        if stat:
            print(self.mstack.stack)
        '''
        
        # Building the tree
        
        # The number of unique values in the dataset will determine the number of branches
        nu = len(self.unique_vals)
        nodes_sides = [x for x in range(0, nu)]
        #print(nodes_sides)
        
        
        track = self.mstack.stack[0]
        # Building the subTree
        
        # Building Level 1 nodes
        lr = []
        for i in range(len(nodes_sides)):
            nd = self.build_temporary_dataframe(self.root.initial, nodes_sides[i], nu)
            self.ndx = nd[0]
            self.ndy = nd[1]
            total_votes = {}
            container = {}
            for fi in feature_initials:
                if fi not in self.features_done0:
                    t = self.get_gini(fi)
                    total_votes[fi], container[fi] = t[0], t[1]
            if len(container)==0:
                break
            #print("--------------Features Gini's-------------------")
            #print(container)
            cont = copy.deepcopy(container)
            new_node = self.get_root(cont)
            #print("New Node : ", new_node)
            node_created = self.build_new_node_with_gini(new_node, cont, total_votes[new_node])
            tn = [node_created]
            track.append(tn)
            lr.append(tn)
            #print(track[0].initial)
            self.features_done0.append(node_created.initial)
        #print(lr)
        #print(track)
        #print(self.mstack.stack)
        
        # Building Nodes from Level 2 till the bottom
        
        ncount = 0
        depth = 0
        references = []
        while(True):
            try:
                ltrack = track[1][0]
                rtrack = track[2][0]
                for z in range(1, len(track)):
                    lr = []
                    feature = ltrack.initial
                    if feature not in self.features_done0:
                        for i in range(len(nodes_sides)):
                            nd = self.build_temporary_dataframe(feature, nodes_sides[i], nu)
                            self.ndx = nd[0]
                            self.ndy = nd[1]
                            total_votes = {}
                            container = {}
                            for fi in feature_initials:
                                if fi not in self.features_done0:
                                    t = self.get_gini(fi)
                                    total_votes[fi], container[fi] = t[0], t[1]
                            if len(container)==0:
                                break
                            #print("--------------Features Gini's-------------------")
                            #print(container)
                            cont = copy.deepcopy(container)
                            new_node = self.get_root(cont)
                            #print("New Node : ", new_node)
                            node_created = self.build_new_node_with_gini(new_node, cont, total_votes[new_node])
                            tn = [node_created]
                            lr.append(tn)
                            #print(track[0].initial)
                            self.features_done0.append(node_created.initial)
                        #print(lr)
                        ltrack[z].append(lr)
                        
                        if len(lr)>1:
                            references.append(lr[1])

                        for r in range(len(references)):
                            feature = references[i].initial
                            if feature not in self.features_done1:
                                for i in range(len(nodes_sides)):
                                    nd = self.build_temporary_dataframe(feature, nodes_sides[i], nu)
                                    self.ndx = nd[0]
                                    self.ndy = nd[1]
                                    total_votes = {}
                                    container = {}
                                    for fi in feature_initials:
                                        if fi not in self.features_done0:
                                            t = self.get_gini(fi)
                                            total_votes[fi], container[fi] = t[0], t[1]
                                    if len(container)==0:
                                        break
                                    #print("--------------Features Gini's-------------------")
                                    #print(container)
                                    cont = copy.deepcopy(container)
                                    new_node = self.get_root(cont)
                                    #print("New Node : ", new_node)
                                    node_created = self.build_new_node_with_gini(new_node, cont, total_votes[new_node])
                                    tn = [node_created]
                                    lr.append(tn)
                                    #print(track[0].initial)
                                    self.features_done0.append(node_created.initial)
                                #print(lr)
                                rtrack[1].append(lr)
                                
                for y in range(1, len(track)):
                    lr = []
                    feature = rtrack.initial
                    if feature not in self.features_done0:
                        for i in range(len(nodes_sides)):
                            nd = self.build_temporary_dataframe(feature, nodes_sides[i], nu)
                            self.ndx = nd[0]
                            self.ndy = nd[1]
                            total_votes = {}
                            container = {}
                            for fi in feature_initials:
                                if fi not in self.features_done0:
                                    t = self.get_gini(fi)
                                    total_votes[fi], container[fi] = t[0], t[1]
                            if len(container)==0:
                                break
                            #print("--------------Features Gini's-------------------")
                            #print(container)
                            cont = copy.deepcopy(container)
                            new_node = self.get_root(cont)
                            #print("New Node : ", new_node)
                            node_created = self.build_new_node_with_gini(new_node, cont, total_votes[new_node])
                            tn = [node_created]
                            lr.append(tn)
                            #print(track[0].initial)
                            self.features_done0.append(node_created.initial)
                        #print(lr)
                        rtrack[y].append(lr)
                        
                        if len(lr)>1:
                            references.append(lr[1])

                        for r in range(len(references)):
                            feature = references[i].initial
                            if feature not in self.features_done1:
                                for i in range(len(nodes_sides)):
                                    nd = self.build_temporary_dataframe(feature, nodes_sides[i], nu)
                                    self.ndx = nd[0]
                                    self.ndy = nd[1]
                                    total_votes = {}
                                    container = {}
                                    for fi in feature_initials:
                                        if fi not in self.features_done0:
                                            t = self.get_gini(fi)
                                            total_votes[fi], container[fi] = t[0], t[1]
                                    if len(container)==0:
                                        break
                                    #print("--------------Features Gini's-------------------")
                                    #print(container)
                                    cont = copy.deepcopy(container)
                                    new_node = self.get_root(cont)
                                    #print("New Node : ", new_node)
                                    node_created = self.build_new_node_with_gini(new_node, cont, total_votes[new_node])
                                    tn = [node_created]
                                    lr.append(tn)
                                    #print(track[0].initial)
                                    self.features_done0.append(node_created.initial)
                                #print(lr)
                                rtrack[len(rtrack)-1].append(lr)
                track = track[1]
            except IndexError:
                break