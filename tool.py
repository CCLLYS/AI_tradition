# 连接到 Milvus 服务器
# "default" 是连接的别名，host 和 port 分别指定服务器的地址和端口。
import os
import json


def embedding_dense(model, question):
    Q_embedding = model.encode(question)['dense_vecs']
    return Q_embedding

# 搜索相似句子Q_A，朴素搜索，仅采用语义相似性
def Q_A(model, collection, question):
    """
    输入：用户输入的问题
    输出：前n条最佳回答
    """
    Q_embedding = embedding_dense(model, question)
    # 匹配参数
    search_params = {
        "metric_type": "COSINE", 
        "params": {"nprobe": 10}  # nprobe 参数
    }

    # 执行搜索，返回n个最匹配的
    results = collection.search([Q_embedding.tolist()], 
                                anns_field = "embedding", 
                                param = search_params, 
                                limit=3,
                                output_fields = ['text','translation','significance','advice','rel2psy','keyword']
                                )
    # 返回结果
    print('----result-----')
    # print(results)
    similar_answer = []
    for result in results[0]:
        query_results = result['entity']
        distance = result['distance']
        similar_answer.append([query_results,distance])
    return similar_answer


# 搜索相似数据，混合检索，先匹配关键词，再匹配语义 Q_A_plus
def Q_A_plus(model, collection, question, emotion):
    """
    输入：嵌入模型model
    输入：用户输入的问题
    输入：用户问题对应的情绪列表
    输出：前n条最佳回答
    """
    print('-----进行数据库搜索-----')
    # 进行向量嵌入
    Q_embedding = embedding_dense(model, question)

    # 进行关键词匹配
    expr = f"ARRAY_CONTAINS_ANY(keyword, {emotion}) "

    # 执行关键词匹配查询
    collection.load()
    keyword_result = collection.query(expr=expr, output_fields=["id", "keyword"])

    # 获取匹配的ID列表
    matched_ids = [item["id"] for item in keyword_result]

    # 如果没有匹配的记录，直接返回空结果
    if not matched_ids:
        print("No matching records found.")

    if matched_ids:
        # 匹配参数
        search_params = {
            "metric_type": "COSINE", 
            "params": {"nprobe": 10}  # nprobe 参数
        }
        # 执行搜索，返回n个最匹配的
        results = collection.search(data = [Q_embedding.tolist()], 
                                    anns_field = "embedding", 
                                    param = search_params, 
                                    limit=3,
                                    expr=expr,
                                    output_fields = ['text','translation','significance','advice','rel2psy','keyword']
                                    )
        # 返回结果
        # print('----result-----')

        # print(results)
        similar_answer = []
        for result in results[0]:
            query_results = result['entity']
            distance = result['distance']
            similar_answer.append([query_results,distance])
    # print(similar_answer)
    return similar_answer


def LLM_response(client, user_prompt): 
    """
    输入：用户查询
    输出：大模型回答
    """
    response = client.chat.completions.create(
        model="glm-4-plus",  # 请填写您要调用的模型名称
        messages=[
            {"role": "system", "content": "你是一个专业的心理咨询师。"},
            {"role": "user", "content": f"{user_prompt}"},
        ],
        stream=False,  # 是否开启流式输出
        temperature = 0.5
    )
    # print('-------------resopnse-----------------')
    # print(response.choices[0].message.content)
    response = response.choices[0].message.content
    return response


# 用户提问改写函数
def rewrite_question(user_prompt_rewrite:str, question:str, client):
    """
    user_prompt_rewrite: 改写的提示词模板
    question: 用户提问
    client: 大模型的代理
    """
    user_prompt_rewrite_temp = user_prompt_rewrite.format(input = question)
    query_temp = LLM_response(client, user_prompt_rewrite_temp)
    query_temp = query_temp.strip()
    print('-----转化后的提问----')
    if "改写内容：" and "情绪色彩：" in query_temp:
        query = query_temp.split('改写内容：')[1].split('情绪色彩：')[0].strip()
        emotion = query_temp.split('改写内容：')[1].split('情绪色彩：')[1]
        emotion = emotion.split('、')
    else:
        print('-----大模型格式错误-----')
    print(query)
    print(emotion)
    return query,emotion
