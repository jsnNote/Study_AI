import numpy as np
import pandas as pd
from pandas import DataFrame


def one_hot_encoding(data, feature=None):
    result = {"columns": [], "encoding": {}}
    columns = set(data)
    encoding_size = len(columns)

    if feature:
        result["columns"] = [f"{feature}_{column}" for column in columns]
    else:
        result["columns"] = columns

    result["encoding"] = dict.fromkeys(columns, [])
    for i, column in enumerate(columns):
        encoding_array = [0] * encoding_size  # 객체의 메모리 할당
        encoding_array[i] = 1  # 해당하는 값에
        result["encoding"][column] = encoding_array
    return result


# 우선순위 데이터 처리 함수
def ordinal_data_preprocessing(df: DataFrame, priority: dict, feature=None):
    rows = []
    for v in df[feature]:
        new_row = [priority[v]]
        rows.append(new_row)
    _df = DataFrame(rows, columns=[f"{feature}_numeric"], index=df.index)
    new_df = pd.concat([df, _df], axis=1)

    return new_df


# 범주형 데이터 처리 함수 - 데이터가 3개 이상인 경우 사용
def categorical_data_preprocessing(df: DataFrame, data: dict, feature=None):
    rows = []
    for v in df[feature]:
        new_row = data["encoding"][v]
        rows.append(new_row)

    _df = DataFrame(rows, columns=data["columns"], index=df.index)
    new_df = pd.concat([df, _df], axis=1)

    return new_df