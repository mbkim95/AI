import numpy as np
import torch
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getDataSet(data_filename):
    data = pd.read_csv(data_filename, sep=';')

    data.loc[data.school=='GP', 'school']=0
    data.loc[data.school=='MS', 'school']=1

    data.loc[data.sex=='M', 'sex'] = 0
    data.loc[data.sex=='F', 'sex'] = 1

    data.loc[data.address=='U', 'address'] = 0
    data.loc[data.address=='R', 'address'] = 1

    data.loc[data.famsize=='LE3', 'famsize'] = 0
    data.loc[data.famsize=='GT3', 'famsize'] = 1

    data.loc[data.Pstatus=='T', 'Pstatus'] = 0
    data.loc[data.Pstatus=='A', 'Pstatus'] = 1

    data.loc[data.Mjob=='teacher', 'Mjob'] = 0
    data.loc[data.Mjob=='health', 'Mjob'] = 1
    data.loc[data.Mjob=='services', 'Mjob'] = 2
    data.loc[data.Mjob=='at_home', 'Mjob'] = 3
    data.loc[data.Mjob=='other', 'Mjob'] = 4

    data.loc[data.Fjob=='teacher', 'Fjob'] = 0
    data.loc[data.Fjob=='health', 'Fjob'] = 1
    data.loc[data.Fjob=='services', 'Fjob'] = 2
    data.loc[data.Fjob=='at_home', 'Fjob'] = 3
    data.loc[data.Fjob=='other', 'Fjob'] = 4

    data.loc[data.reason=='home', 'reason'] = 0
    data.loc[data.reason=='reputation', 'reason'] = 1
    data.loc[data.reason=='course', 'reason'] = 2
    data.loc[data.reason=='other', 'reason'] = 3

    data.loc[data.guardian=='mother', 'guardian'] = 0
    data.loc[data.guardian=='father', 'guardian'] = 1
    data.loc[data.guardian=='other', 'guardian'] = 2

    data.loc[data.schoolsup=='yes', 'schoolsup'] = 1
    data.loc[data.schoolsup=='no', 'schoolsup'] = 0

    data.loc[data.famsup=='yes', 'famsup'] = 1
    data.loc[data.famsup=='no', 'famsup'] = 0

    data.loc[data.paid=='yes', 'paid'] = 1
    data.loc[data.paid=='no', 'paid'] = 0

    data.loc[data.activities=='yes', 'activities'] = 1
    data.loc[data.activities=='no', 'activities'] = 0

    data.loc[data.nursery=='yes', 'nursery'] = 1
    data.loc[data.nursery=='no', 'nursery'] = 0

    data.loc[data.higher=='yes', 'higher'] = 1
    data.loc[data.higher=='no', 'higher'] = 0

    data.loc[data.internet=='yes', 'internet'] = 1
    data.loc[data.internet=='no', 'internet'] = 0

    data.loc[data.romantic=='yes', 'romantic'] = 1
    data.loc[data.romantic=='no', 'romantic'] = 0
    return getInputData(torch.tensor(data.values)), getTargetData(torch.tensor(data.values)).long(), getInputData(torch.tensor(data.values)).shape[1]

def getInputData(data):
    ret = data[:, :-1]
    return ret

def getTargetData(data):
    target = data[:,-1]
    return target

def preprocess(data):
    input_matrix, target_matrix, features_counts = getDataSet(data)

    input_matrix = torch.t(input_matrix).numpy()
    input_matrix = np.array(input_matrix, dtype=np.float32)

    for i, j in enumerate(input_matrix):
        j = normalize(j)
        j = standardize(j)
        input_matrix[i] = j

    input_matrix = torch.from_numpy(input_matrix)
    input_matrix = torch.t(input_matrix)

    return input_matrix, target_matrix, features_counts

def normalize(data):                                    # 데이터를 0 ~ 1 사이 값으로 바꾸는 것
    min_data, max_data = data.min(), data.max()
    if min_data == max_data:
        if max_data >= 1:
            return np.array([1. for _ in range(len(data))])
        else:
            return 0
    data = (data - min_data) / (max_data - min_data)
    data.round()
    return data


def standardize(data):                                  # 각 값들을 표준편차로 바꿈
    means = np.mean(data)
    stds = np.std(data)
    if stds == 0:
        stds = 1
    standardized_data = (data - means) / stds
    return standardized_data

def cross_validation(ratio, input_data, target_data):                        # 교차검증
    input_arrays, target_arrays, length = [], [], len(input_data)
    quo = length // ratio

    # Shuffle data
    # indexes = [idx for idx in range(input_data.size()[0])]
    # random.shuffle(indexes)
    # input_data = input_data[indexes]
    # target_data = target_data[indexes]
    for idx in range(ratio):
        start = idx * quo
        end = (idx + 1) * quo if idx != ratio - 1 else (idx + 1) * quo + length % ratio
        input_arrays.append(input_data[start:end])
        target_arrays.append(target_data[start:end])
    # return [input_arrays[idx] for idx in indexes], [target_arrays[idx] for idx in indexes]
    return input_arrays, target_arrays

input_matrix, target_matrix, features_counts = getDataSet("./student/student-mat.csv")
