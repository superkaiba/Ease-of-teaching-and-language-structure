# Generate data
import numpy as np
import random
import itertools
import pdb
import torch
class Dataset2:
    def __init__(self, numColors, numShapes, attrSize, args):
        self.numColors = numColors  # number of colors appeared in the dataset
        self.numShapes = numShapes  # number of shapes appeared in the dataset
        self.attrSize = attrSize  # bits of features
        self.train_np = np.array([])

    def getTrain(self): 
        trainInd = [(i, j) for i in range(self.numColors) for j in range(self.numShapes)]

        train_np = np.zeros((len(trainInd), 2), dtype=int)
        for ind in range(len(trainInd)):
            train_np[ind] = [trainInd[ind][0], trainInd[ind][1]]
        self.train_np = train_np
        return train_np

    def getBatchData(self, indices, batch, distractNum):
        # We need batch because we generate different instances consisting of all the distractors and target within
        # sample train batch from data
        # return numpy array [batch, attrLength]
        color = np.zeros([batch, distractNum, self.numColors], dtype=np.float32)
        shape = np.zeros([batch, distractNum, self.numShapes], dtype=np.float32)
        numTuples = len(indices) # number of tuples in the training set
        batchInd = [random.sample(range(numTuples), distractNum) for _ in range(batch)] # non-repetitive
        # fetch the batchid and turn color/shape index into one hot nunpy vertor
        for i in range(batch):
            for j in range(distractNum):
                colorindex = indices[batchInd[i][j]][0]
                shapeindex = indices[batchInd[i][j]][1]
                color[i][j][colorindex] = 1
                shape[i][j][shapeindex] = 1
        if self.attrSize != self.numColors + self.numShapes:
            x_coordinate = np.random.rand(batch, distractNum, 1)
            y_coordinate = np.random.rand(batch, distractNum, 1)
            instances = np.concatenate([color, shape, x_coordinate, y_coordinate], axis=2)
        else:
            instances = np.concatenate([color, shape], axis=2)
        # extract one target from distract tuples
        targetInd = np.random.randint(distractNum, size=(batch), dtype=int)
        targets = instances[np.arange(batch), targetInd, :]
        return instances, targets #(batch, distract, attrSize) (batch, attrSize)

    def getEnumerateData(self):
        attrVector = np.zeros([self.numColors * self.numShapes, self.numColors + self.numShapes], dtype=np.float32)
        for i in range(self.numColors):
            for j in range(self.numShapes):
                attrVector[i * self.numShapes + j][i] = 1
                attrVector[i * self.numShapes + j][self.numColors + j] = 1
        return attrVector

class Dataset:
    def __init__(self, args):
        self.args = args
        self.all_combos = None

    def getBatchData(self, a1, a2, a3):
        # We need batch because we generate different instances consisting of all the distractors and target within
        # sample train batch from data
        # return numpy array [batch, attrLength]
        batch_size = self.args["batchSize"]
        distract_num = self.args["distractNum"]

        data = np.zeros([batch_size, distract_num, self.args["n_values"] * self.args["n_attributes"]], dtype=np.float32)
        for i in range(batch_size):
            for j in range(distract_num):
                for k in range(self.args["n_attributes"]):
                    l = np.random.randint(self.args["n_values"])
                    data[i][j][k * self.args["n_values"] + l] = 1      

        # extract one target from distract tuples
        targetInd = np.random.randint(distract_num, size=(batch_size), dtype=int)
        targets = data[np.arange(batch_size), targetInd, :]
        return data, targets, torch.tensor(targetInd, device="cuda") #(batch, distract, attrSize) (batch, attrSize)

    def getEnumerateDataIdxes(self):
        lists = [np.arange(self.args["n_values"]) for i in range(self.args["n_attributes"])]
        idxes = np.array(list(itertools.product(*lists)))
        return idxes

    def getEnumerateData(self):
        if self.all_combos is None:
            self.all_combos = np.zeros([self.args["n_values"] ** self.args["n_attributes"], self.args["n_attributes"] * self.args["n_values"]], dtype=np.float32)
            idxes = self.getEnumerateDataIdxes()

            for i, idx in enumerate(idxes):
                for j, val in enumerate(idx):
                    self.all_combos[i][j * self.args["n_values"] + val] = 1
        return self.all_combos
