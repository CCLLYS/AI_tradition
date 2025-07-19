'''本文档用于数据构建完之后，将数据存入数据库中'''
'''这里的数据库选择为milvus，利用docker部署，注意环境'''

'''
数据结构
0-6,embedding
原文、注释、翻译、rel2psy、significance、advice、keyword、embedding
'''

from pymilvus import connections
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import os
import json
import pandas as pd

# 检查缺失值
def find_isnull(data):
    # 检查整个 DataFrame 的缺失值
    missing_values_total = pd.DataFrame(data).isnull().sum()

    # 检查每一行的缺失值
    missing_values_per_row = pd.DataFrame(data).isnull().sum(axis=1)

    # 打印出每行缺失值的数量
    for index, missing_count in missing_values_per_row.items():
        if missing_count > 0:
            print(f"Row {index} has {missing_count} missing values.")

    # 如果你想找出具体哪些列缺失，可以这样做：
    missing_value_details = pd.DataFrame(data).isnull()

    # 打印每行缺失值的具体信息
    for index, row in missing_value_details.iterrows():
        missing_columns = row[row].index.tolist()
        if missing_columns:
            print(f"Row {index} has missing values in columns: {missing_columns}")
    return missing_values_total



if __name__ == '__main__':
    # 数据文件夹或者单个文件嵌入
    # file_path = r"D:\python_code\Aitradition\answer_book\data\【after_embedding】道德经.xlsx"
    folder_path = r'D:\python_code\answer_book\data\milvus_final'

    # 连接到 Milvus 服务器
    # "default" 是连接的别名，host 和 port 分别指定服务器的地址和端口。
    connections.connect("default", host="localhost", port="19530")

    # 定义字段
    # 使用 FieldSchema 定义集合中的字段，包括字段名、数据类型和其他属性。
    """
    id:编号，整数编码
    text:原文，字符串
    note:注释，字符串
    translation:译文，字符串
    rel2psy:心理学关联内容，字符串
    significance:现实意义与启发，字符串
    embedding:嵌入向量，向量
    advice:解决路径及方法，字符串
    keyword:关键词列表，数组

    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="note", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="translation", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="rel2psy", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="significance", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="advice", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="keyword", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length = 1024, max_capacity=10)
    ]

    # 定义集合模式
    # 使用 CollectionSchema 定义集合的模式，包括字段和描述。
    # description：对 Collection 的简要描述（可选）。
    schema = CollectionSchema(fields, description="论语和道德经数据合集")
    # 创建集合
    # 使用 Collection 创建集合。
    # 数据库的名称为：RAG_embedding
    collection = Collection(name="guoxue", schema=schema)


    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        # 对字段重命名
        """
        text:原文，字符串
        note:注释，字符串
        translation:译文，字符串
        rel2psy:心理学关联内容，字符串
        significance:现实意义与启发，字符串
        embedding:嵌入向量，向量
        advice:解决路径及方法，字符串
        keyword:关键词列表，数组
        """
        data = pd.read_excel(file_path)
        renamed_columns = data.rename(columns={0: 'text',
                                            1: 'note',
                                            2: 'translation',
                                            3: 'rel2psy',
                                            4: 'significance',
                                            5: 'advice',
                                            6: 'keyword',
                                            'embedding': 'embedding'})
        data = renamed_columns.to_dict('records')

        # 修改嵌入和关键词列表为list格式
        for i in range(len(data)):
            data[i]['embedding'] = json.loads(data[i]['embedding'])
            data[i]['keyword'] = data[i]['keyword'].split(',')

        mr = collection.insert(data)
        collection.flush()

    # 创建索引
    index_params = {
        "metric_type": "COSINE",  # 距离度量类型
        "index_type": "FLAT",  # 索引类型
    }

    collection.create_index(field_name="embedding", index_params=index_params)
    # 启用
    collection.load()

    print('数据库加载完成')



