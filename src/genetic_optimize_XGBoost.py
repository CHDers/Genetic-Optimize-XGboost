# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 9:06
# @Author  : Yanjun Hao
# @Site    : 
# @File    : genetic_optimize_XGBoost.py
# @Software: PyCharm 
# @Comment :

# NOTE: Linking:
#  使用遗传算法在 XGBoost 中调整超参数: https://mp.weixin.qq.com/s/vtuYBKzc4xpv2LUMt6_86w
#  xgboost的遗传算法调参: https://www.cnblogs.com/wzdLY/p/9700574.html

import numpy as np
import random
from sklearn.metrics import f1_score
import xgboost


class GeneticXgboost:
    def __init__(self, num_parents=None):
        """
        param num_parents:种群个体的数量

        """
        self.num_parents = num_parents

    def initilialize_poplulation(self):
        """
        初始化种群,即生成规定数量的种群的基因
        learning_rate,n_estimators，max_depth,min_child_weightsubsample,olsample_bytree,gamma
        return：array,shape=[self.num_parents,num_gene]
        """
        learningRate = np.empty([self.num_parents, 1])
        nEstimators = np.empty([self.num_parents, 1], dtype=np.uint8)
        maxDepth = np.empty([self.num_parents, 1], dtype=np.uint8)
        minChildWeight = np.empty([self.num_parents, 1])
        gammaValue = np.empty([self.num_parents, 1])
        subSample = np.empty([self.num_parents, 1])
        colSampleByTree = np.empty([self.num_parents, 1])
        for i in range(self.num_parents):
            # 生成每个个体
            learningRate[i] = round(np.random.uniform(0.01, 1), 2)
            nEstimators[i] = int(random.randrange(10, 1500, step=25))
            maxDepth[i] = int(random.randrange(1, 10, step=1))
            minChildWeight[i] = round(random.uniform(0.01, 10.0), 2)
            gammaValue[i] = round(random.uniform(0.01, 10.0), 2)
            subSample[i] = round(random.uniform(0.01, 1.0), 2)
            colSampleByTree[i] = round(random.uniform(0.01, 1.0), 2)
        population = np.concatenate((learningRate, nEstimators, maxDepth, minChildWeight,
                                     gammaValue, subSample, colSampleByTree), axis=1)
        return population

    def fitness_function(self, y_true, y_pred):
        """
        定义适应度函数
        """
        fitness = round((f1_score(y_true, y_pred, average='weighted')), 4)
        return fitness

    def fitness_compute(self, population, dMatrixTrain, dMatrixtest, y_test):
        """
        计算适应度值
        param population:  种群
        param dMatrixTrain:训练数据，（X,y)
        param dMatrixtest: 测试数据, (x,y)
        param y_test:      测试数据y
        return 种群中每个个体的适应度值
        """
        f1_Score = []
        for i in range(population.shape[0]):  # 遍历种群中的每一个个体
            param = {'objective': 'binary:logistic',
                     'learning_rate': population[i][0],
                     'n_estimators': population[i][1],
                     'max_depth': int(population[i][2]),
                     'min_child_weight': population[i][3],
                     'gamma': population[i][4],
                     'subsample': population[i][5],
                     'colsample_bytree': population[i][6],
                     'seed': 24}
            num_round = 100
            model = xgboost.train(param, dMatrixTrain, num_round)
            preds = model.predict(dMatrixtest)
            preds = preds > 0.5
            f1 = self.fitness_function(y_test, preds)
            f1_Score.append(f1)
        return f1_Score

    def parents_selection(self, population, fitness, num_store):
        """
        根据适应度值来选择保留种群中的个体数量
        param population:种群，shape=[self.num_parents,num_gene]
        param num_store: 需要保留的个体数量
        param fitness:   适应度值，array
        return 种群中保留的最好个体，shape=[num_store,num_gene]
        """
        # 用于存储需要保留的个体
        selectedParents = np.empty((num_store, population.shape[1]))
        for parentId in range(num_store):
            # 找到最大值的索引
            bestFitnessId = np.where(fitness == np.max(fitness))
            bestFitnessId = bestFitnessId[0][0]
            # 保存对应的个体基因
            selectedParents[parentId, :] = population[bestFitnessId, :]
            # 将提取了值的最大适应度赋值-1，避免再次提取到
            fitness[bestFitnessId] = -1

        return selectedParents

    def crossover_uniform(self, parents, childrenSize):
        """
        交叉
        我们使用均匀交叉，其中孩子的每个参数将基于特定分布从父母中独立地选择
        param parents:
        param childrenSize:
        return
        """

        crossoverPointIndex = np.arange(0, np.uint8(childrenSize[1]), 1, dtype=np.uint8)
        crossoverPointIndex1 = np.random.randint(0, np.uint8(childrenSize[1]),
                                                 np.uint8(childrenSize[1] / 2))
        crossoverPointIndex2 = np.array(list(set(crossoverPointIndex) - set(crossoverPointIndex1)))
        children = np.empty(childrenSize)

        # 将两个父代个体进行交叉
        for i in range(childrenSize[0]):
            # find parent1 index
            parent1_index = i % parents.shape[0]
            # find parent 2 index
            parent2_index = (i + 1) % parents.shape[0]
            # insert parameters based on random selected indexes in parent1
            children[i, crossoverPointIndex1] = parents[parent1_index, crossoverPointIndex1]
            # insert parameters based on random selected indexes in parent1
            children[i, crossoverPointIndex2] = parents[parent2_index, crossoverPointIndex2]
        return children

    def mutation(self, crossover, num_param):
        '''
        突变
        随机选择一个参数并通过随机量改变值来引入子代的多样性
        param crossover:要进行突变的种群
        param num_param:参数的个数
        return
        '''

        # 定义每个参数允许的最小值和最大值
        minMaxValue = np.zeros((num_param, 2))

        minMaxValue[0, :] = [0.01, 1.0]  # min/max learning rate
        minMaxValue[1, :] = [10, 2000]  # min/max n_estimator
        minMaxValue[2, :] = [1, 15]  # min/max depth
        minMaxValue[3, :] = [0, 10.0]  # min/max child_weight
        minMaxValue[4, :] = [0.01, 10.0]  # min/max gamma
        minMaxValue[5, :] = [0.01, 1.0]  # min/maxsubsample
        minMaxValue[6, :] = [0.01, 1.0]  # min/maxcolsample_bytree

        # 突变随机改变每个后代中的单个基因
        mutationValue = 0
        parameterSelect = np.random.randint(0, 7, 1)

        if parameterSelect == 0:  # learning_rate
            mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
        if parameterSelect == 1:   # n_estimators
            mutationValue = np.random.randint(-200, 200, 1)
        if parameterSelect == 2:  # max_depth
            mutationValue = np.random.randint(-5, 5, 1)
        if parameterSelect == 3:  # min_child_weight
            mutationValue = round(np.random.uniform(5, 5), 2)
        if parameterSelect == 4:  # gamma
            mutationValue = round(np.random.uniform(-2, 2), 2)
        if parameterSelect == 5:  # subsample
            mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
        if parameterSelect == 6:  # colsample
            mutationValue = round(np.random.uniform(-0.5, 0.5), 2)

        # 通过更改一个参数来引入变异，如果超出范围则设置为max或min
        for idx in range(crossover.shape[0]):
            crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue

            if (crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
                crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 1]

            if (crossover[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):
                crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 0]

        return crossover
