import gradio as gr
from pymilvus import connections
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from FlagEmbedding import BGEM3FlagModel
from zhipuai import ZhipuAI
import os
from logger_config import logger
import asyncio

from tool import Q_A_plus, LLM_response, rewrite_question, Q_A

# 连接数据库
try:
    connections.connect("default", host="localhost", port="19530")
    # 启用数据库collection
    collection = Collection(name="guoxue")
    collection.load()
except Exception as e:
    logger.info(e)

# 加载嵌入模型model
try:
    # model = BGEM3FlagModel(r'D:\python_code\answer_book\model\local_bge_m3', use_fp16=True)
    model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
except Exception as e:
    logger.info(e)

# 设置大模型key
api_key = os.getenv('ZHIPU_API_KEY')
client = ZhipuAI(api_key=api_key)

# 全局变量
search_milvus_content = []

# 改写提示词
user_prompt_rewrite = """
请扮演一个擅长理解和转述情绪的助手。你的任务是将一段表达不确定或简单的陈述，改写成一个更具体、更能反映当前处境、情绪状态以及对未来可能结果的感受的句子。

原始输入：

{input}

改写要求：

1. 识别核心不确定性： 理解原始句子中隐含的对某几件事能否成功的疑问或不确定感。
2. 加入情绪色彩： **从以下20个词语里面选择5个最匹配的情绪**：快乐、悲伤、愤怒、恐惧、惊讶、厌恶、焦虑、兴奋、迷茫、期待、失落、疲惫、自豪、郁闷、羡慕、自我怀疑、羞耻、内疚、思念、害怕。
3. 明确表达需求： 将原始的疑问或不确定感，转变为对现状或未来的担忧或害怕，或者是对未来的期待。
4. 保持自然流畅： 确保改写后的句子听起来自然、连贯，像是当事人真实的心声，符合口语表达习惯，但同时回答要避免啰嗦重复。

请将以下原始句子按照上述要求进行改写，并输出改写后的句子和情绪色彩，情绪色彩以”、”进行分割，且**需要从给出的20个情绪色彩词语中进行选择**。直接输出以下内容：

改写内容：[content]

情绪色彩：[emotion]
"""

# 大模型最终回答提示词
user_prompt_final = """
请扮演一位既懂传统文化又理解现代心理学的心理学顾问。你的任务是根据用户的问题和我们找到的相关论语原文（text）及其信息（“rel2psy”、“advice”），生成一段积极且有指导意义的回复给用户。

具体要求：

1. 整合信息： 从原文及其信息中，总结出最契合用户当前情绪和困境的核心观点。
2. 心理连接： 简要提及“rel2psy”中提到的心理学理论，解释古训观点与现代心理学在应对用户情绪上的相通之处，重点突出积极面或应对策略。
3. 给出建议： 主要依据“advice”部分，为用户提供1-2条具体、可行、积极的心理调适方法或行动建议，并可以简单说明这样做可能带来的心理益处。
4. 控制长度： 回答要精炼，避免冗长，确保用户能快速抓住要点，感到被理解并获得力量。
5. 基调： 保持积极、共情、鼓励的语气。
6. 回答中不需要提及古训，模拟心理咨询的场景口语化地回应。需要结合现代心里理论给出启发性指导。

输入：

用户问题：{query}

相关论语原文及信息：

{message}

输出：

请直接输出一段**积极、有现实指导意义、情绪上能带来鼓励**的回应。**语言自然、有温度，避免说教**，要体现中国文化的智慧，也要兼顾现代心理科学的理性。
"""


# 输入完问题和翻页的页数的时候点击按钮调用的函数
def flip_page(page_num, question, flip_counter):
    """
    处理翻页逻辑
    page_num是当前选择的页数
    question是用户的问题
    flip_counter是该问题下的翻页次数
    """
    
    if not page_num or not question:
        return "⚠️ 页码或问题不能为空！", "无法获取答案", flip_counter
    try:
        page = int(page_num)
        # 修改返回格式，明确显示输入的页码
    except ValueError:
        return "⚠️ 请输入有效的数字页码或问题！", "无法获取答案", flip_counter
    
    # 初始化问题翻页次数
    if question not in flip_counter:
        flip_counter[question] = 0

    # 判断是否超过次数
    if flip_counter[question] >= 3:
        return (
            f"🚫 已超过最大翻页次数（3次）",
            f"请认真思考你的问题：{question}\n你已经查看过3次相关答案。",
            flip_counter
        )

    # 增加当前问题的翻页次数
    flip_counter[question] += 1

    # 处理搜索逻辑
    print('-----进行数据库搜索-----')

    # 如果是第一次调用
    if flip_counter[question] == 1:
        # gr.Info("加载中，请稍后...")
        try:
            # 首先进行改写
            query, emotion = rewrite_question(user_prompt_rewrite, question, client)

            # 进行数据库检索
            # response = Q_A_plus(model, collection, query, emotion)
            # print(response)
            # 或者
            response = Q_A(model, collection, query)
            print(response)
            # [[{'rel2psy':'','keyword':'','text':'','translation':'','significance':'','advice':''},0.98],...[]]
            global search_milvus_content
            search_milvus_content = response

            # 获取第一条
            data_now = search_milvus_content[0][0]
            text_now = data_now['text']
            print("当前的原文是",text_now)

            return (
                f"📖 当前页码：{page_num}（第 {flip_counter[question]} 次翻页）", 
                text_now,
                flip_counter
            )
        
        except Exception as e:
            print(e)
            return (f"📖 当前页码：{page_num}（第 {flip_counter[question]} 次翻页）",
                    '调用大模型出错，无法获取答案', 
                    flip_counter)
    
    else:
        # 获取当前的原文，与翻页次数有关
        data_now = search_milvus_content[flip_counter[question]-1][0]
        text_now = data_now['text']
        print("当前的原文是",text_now)
        return (
                f"📖 当前页码：{page_num}（第 {flip_counter[question]} 次翻页）", 
                text_now,
                flip_counter
            )


def show_analysis(text, question):
    """根据答案生成解析"""
    if "无法获取答案" in text:
        return "无法为无效页码提供解析"

    # 返回该原文的解析
    # 首先在全局变量里面找到对应的数据
    global search_milvus_content
    for data in search_milvus_content:
        # [{'rel2psy':'','keyword':'','text':'','translation':'','significance':'','advice':''},0.98]
        data = data[0]
        if text == data['text']:
            # 找到数据，输出解析，并且调用大模型给回答
            translation = data['translation']
            print('---大模型回答---')
            try:
                message = {key: data[key] for key in ['text','rel2psy','advice']}
                user_prompt_final_temp = user_prompt_final.format(query = question, message = message)
                LLM_answer = LLM_response(client ,user_prompt_final_temp)
                print(LLM_answer)
                # LLM_answer = LLM_answer.replace('\n\n','\n')
                return f"这句话的译文为：{translation}\n\n{LLM_answer}"
            except Exception as e:
                print(e)
                return f"译文为：{translation}"
        else:
            print('找不到对应的原文')
            continue

    return "找不到对应的解析"

# 自定义CSS样式
css = """
/* 设置所有标签（label）的字号 */
label {
    font-size: 16px !important;  /* 你可以根据需要调整大小，例如16px, 18px等 */
}
"""

with gr.Blocks(title="答案之书", css=css) as demo:
    # 标题和问题输入
    gr.Markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 24px; font-weight: bold;">
            📖 答案之书 📖
        </div>
        <div style="font-size: 16px; margin-top: 8px;">
            随机抽取一句古语，为当下的你提供思考线索。所得句子来自传统典籍《论语》《道德经》。
        </div>
        <div style="font-size: 16px; color: #e74c3c; margin-top: 4px;">
            每次使用前，心中默念一个你此刻想问的问题。
        </div>
    </div>
    """)

    # 问题输入框
    question_input = gr.Textbox(
        placeholder="写下你的困惑...",
        label="你的问题❓",
        lines=2
    )

    # 页码输入框（带emoji提示）
    page_input = gr.Textbox(
        label= "你想要翻到的页数 🔢(请从1-521之间选择)",
        info = '⚠️每个问题仅有三次翻页机会❗',
        placeholder = "请输入数字页码（例如：520）"
    )

    # 翻页按钮
    flip_btn = gr.Button("翻页", variant="primary")
    loading_msg1 = gr.Markdown("<div style='color: #666; text-align: center;'>████████████████████⏳ 正在生成答案，请稍候...████████████████████</div>", visible=False)

    # 输出组件
    page_result = gr.Textbox(
        label="页码确认",  # 修改为更明确的标签
        visible=False,
        text_align="center"
    )
    answer_output = gr.Textbox(
        label="你的答案",  # 修改为更生动的标签
        visible=False,
        lines=2,
        text_align="center"
    )

    # 新增解析按钮和解析文本框
    analysis_btn = gr.Button("查看解析", visible=False)
    loading_msg2 = gr.Markdown("<div style='color: #666; text-align: center;'>████████████████████⏳ 正在生成解析，请稍候...████████████████████</div>", visible=False)
    analysis_output = gr.Textbox(
        label="🔍 深度解析",
        visible=False,
        lines=15
    )
    flip_counter = gr.State({})  # 存储问题翻页次数的状态

    # 事件处理
    flip_btn.click(
        fn=lambda: gr.Markdown(visible=True),
        outputs=[loading_msg1]
    ).then(
        fn=flip_page,
        inputs=[page_input, question_input, flip_counter],
        outputs=[page_result, answer_output, flip_counter]
    ).then(
        lambda: [
            gr.Markdown(visible=False),
            gr.Textbox(visible=True), 
            gr.Textbox(visible=True), 
            gr.Button(visible=True)
            ],
        outputs=[loading_msg1, page_result, answer_output, analysis_btn]
    )

    # 解析按钮点击事件
    analysis_btn.click(
        fn=lambda: gr.Markdown(visible=True),
        outputs=[loading_msg2]
    ).then(
        fn=show_analysis,
        inputs=[answer_output, question_input],
        outputs=[analysis_output]
    ).then(
        lambda: [
            gr.Markdown(visible=False),
            gr.Textbox(visible=True)
        ],
        outputs=[loading_msg2, analysis_output]
    )

demo.launch(share=True)