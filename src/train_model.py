# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 9:20
# @Author  : Yanjun Hao
# @Site    : 
# @File    : train_model.py
# @Software: PyCharm 
# @Comment :


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
import random
from sklearn.metrics import f1_score
import xgboost
from rich import print
from tqdm import tqdm
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT_0 = FILE.parents[0]  # project root directory
ROOT_1 = FILE.parents[1]  # project root directory
if str(ROOT_0) not in sys.path:
    sys.path.append(str(ROOT_0))  # add ROOT to PATH
if str(ROOT_1) not in sys.path:
    sys.path.append(str(ROOT_1))  # add ROOT to PATH

from src.genetic_optimize_XGBoost import GeneticXgboost

warnings.filterwarnings('ignore')


def main():
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    xgDMatrixTrain = xgboost.DMatrix(X_train, y_train)
    xgbDMatrixTest = xgboost.DMatrix(X_test, y_test)

    number_of_parents = 8  # 初始种群数量
    number_of_generations = 100  # 种群繁殖代数，即迭代次数
    number_of_parameters = 7  # 将被优化的参数数量
    number_of_parents_mating = 4  # 每代被保留的个体数量

    gx = GeneticXgboost(num_parents=number_of_parents)

    # 定义种群的大小
    populationSize = (number_of_parents, number_of_parameters)

    # 初始种群
    population = gx.initilialize_poplulation()
    # 定义一个数组来存储fitness历史
    FitnessHistory = np.empty([number_of_generations + 1, number_of_parents])
    # 定义一个数组来存储每个父节点和生成的每个参数的值
    populationHistory = np.empty([(number_of_generations + 1) * number_of_parents,
                                  number_of_parameters])
    # 历史记录中插入初始参数的值
    populationHistory[0:number_of_parents, :] = population

    # 训练
    for generation in tqdm(range(number_of_generations)):
        print("This is number %s generation" % (generation))
        # train the dataset and obtain fitness
        FitnessValue = gx.fitness_compute(population=population,
                                          dMatrixTrain=xgDMatrixTrain,
                                          dMatrixtest=xgbDMatrixTest,
                                          y_test=y_test)

        FitnessHistory[generation, :] = FitnessValue
        print('Best F1 score in the iteration = {}'.format(np.max(FitnessHistory[generation, :])))
        # 保留的父代
        parents = gx.parents_selection(population=population,
                                       fitness=FitnessValue,
                                       num_store=number_of_parents_mating)
        # 生成的子代
        children = gx.crossover_uniform(parents=parents,
                                        childrenSize=(populationSize[0] - parents.shape[0], number_of_parameters))

        # 增加突变以创造遗传多样性
        children_mutated = gx.mutation(children, number_of_parameters)

        # 创建新的种群，其中将包含以前根据fitness value选择的父代，和生成的子代
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = children_mutated
        populationHistory[(generation + 1) * number_of_parents:(generation + 1) * number_of_parents + number_of_parents,
        :] = population

    # 最终迭代的最佳解决方案
    fitness = gx.fitness_compute(population=population,
                                 dMatrixTrain=xgDMatrixTrain,
                                 dMatrixtest=xgbDMatrixTest,
                                 y_test=y_test)

    bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]
    print("Best fitness is =", fitness[bestFitnessIndex])
    print("Best parameters are:")
    print('learning_rate=', population[bestFitnessIndex][0])
    print('n_estimators=', population[bestFitnessIndex][1])
    print('max_depth=', int(population[bestFitnessIndex][2]))
    print('min_child_weight=', population[bestFitnessIndex][3])
    print('gamma=', population[bestFitnessIndex][4])
    print('subsample=', population[bestFitnessIndex][5])
    print('colsample_bytree=', population[bestFitnessIndex][6])


if __name__ == '__main__':
    main()
