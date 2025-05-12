import os
from dotenv import load_dotenv
from ai71 import AI71
from chatclient import FalconChatClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datasets import load_dataset

def test_linkage():
    # 加载环境变量
    load_dotenv()
    api_key = os.getenv("AI71_API_KEY")

    # 初始化 AI71 客户端
    client = AI71(api_key=api_key)

    # 初始化对话消息
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # 添加用户提问
    user_input = "你对 AI 技术怎么看？"
    messages.append({"role": "user", "content": user_input})

    # 打印用户提问
    print(f"User: {user_input}")
    print("Falcon:", end=" ", flush=True)

    # 初始化助手回复内容
    assistant_reply = ""

    # 调用模型并流式输出回复
    for chunk in client.chat.completions.create(
        messages=messages,
        model="tiiuae/falcon3-10b-instruct",
        stream=True,
    ):
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            print(delta_content, end="", flush=True)
            assistant_reply += delta_content

    # 添加助手回复到消息列表
    messages.append({"role": "assistant", "content": assistant_reply})
    print("\n")



def genearte_test_questions(que_num=10, step_num=2):
    load_dotenv()
    api_key = os.getenv("AI71_API_KEY")
    
    

    client = FalconChatClient(api_key = api_key)
    question_list = []
    # 定义提示词
    prompt = "请随机生成一个测试问题,不必给出解答"

    despite_text = "目前已有的问题包括："

    # for step_flag in range(step_num):
    #     this_round = []
    for que_flag in range(que_num):
        # 调用 Falcon 模型生成问题
        response = client.single_turn(prompt)
        # 输出生成的问题
        qustion_con = response
        print(qustion_con)
        question_list.append(qustion_con)

    return question_list
    



    

    




def main():
    load_dotenv()
    api_key = os.getenv("AI71_API_KEY")
    
    

    client = FalconChatClient(api_key = api_key)
    # prompts = [
    #     "你对 AI 技术怎么看？",
    #     "请解释一下量子计算的基本原理。",
    #     "如何提高机器学习模型的准确性？",
    #     "什么是区块链技术？",
    #     "介绍一下深度学习的发展历程。"
    # ]
    # ##################### 无背景 串行/并行 14-3
    # # 串行调用
    # start_time = time.perf_counter()
    # for prompt in prompts:
    #     response = client.single_turn(prompt)
    #     print(f"Prompt: {prompt}\nResponse: {response}\n")
    # serial_duration = time.perf_counter() - start_time
    # print(f"串行执行时间: {serial_duration:.2f} 秒\n")

    # # 并行调用
    # start_time = time.perf_counter()
    # results = client.batch_single_turn(prompts, max_workers=5)
    # for prompt, response in results:
    #     print(f"Prompt: {prompt}\nResponse: {response}\n")
    # parallel_duration = time.perf_counter() - start_time
    # print(f"并行执行时间: {parallel_duration:.2f} 秒")



    # ################### 有背景 串行/并行
    # # 准备测试数据
    # user_inputs = genearte_test_questions()
    # histories = [[] for _ in range(10)]

    # # 串行处理
    # start_time = time.perf_counter()
    # serial_results = []
    # for history, user_input in zip(histories, user_inputs):
    #     result = client.multi_turn(history, user_input)
    #     serial_results.append(result)
    # end_time = time.perf_counter()
    # print(f"串行处理耗时: {end_time - start_time:.2f} 秒")

    # # 并行处理
    # start_time = time.perf_counter()
    # parallel_results = client.batch_multi_turn(histories, user_inputs, max_workers=8)
    # end_time = time.perf_counter()
    # print(f"并行处理耗时: {end_time - start_time:.2f} 秒")

    print("main function ended")






def build_index():
    # 加载 sample-10BT 全量数据（流式或一次加载）
    ds = load_dataset("HuggingFaceFW/fineweb",
                      name="sample-10BT",
                      split="train",
                      streaming=False)  # 或 True 并用批次收集
    texts = [ex["text"] for ex in ds]  # 全样本 text 列表
    ids   = [ex["id"]   for ex in ds]  # 全样本 id 列表

    # 嵌入模型
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeds = model.encode(texts, show_progress_bar=True)

    # 建索引
    dim   = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeds, dtype='float32'))

    # 保存
    faiss.write_index(index, "fineweb_10bt.index")
    with open("fineweb_10bt_meta.pkl", "wb") as f:
        pickle.dump({"texts": texts, "ids": ids}, f)

def retrieve(query, k=5):
    # 加载 index & metadata
    index_data = faiss.read_index("fineweb_10bt.index")
    with open("fineweb_10bt_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 查询嵌入与搜索
    q_emb = model.encode([query])
    D, I = index_data.search(np.array(q_emb, dtype='float32'), k)
    results = []
    for idx in I[0]:
        results.append({
            "passage": meta["texts"][idx],
            "doc_IDs": [meta["ids"][idx]]
        })
    return results

if __name__ == "__main__":
    # build_index()
    # res = retrieve("What are benefits of AI?")
    # print(res)
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)

    # 打印数据集中第一项
    print(ds[0])

# if __name__ == "__main__":
#     from datasets import load_dataset
#     from datatrove.pipeline.readers import ParquetReader

#     streaming_ds = load_dataset(
#         "HuggingFaceFW/fineweb",
#         name="sample-10BT",
#         split="train",
#         streaming=True,      # 流式读取
#     )
#     for example in streaming_ds:
#         # 每次只在内存中保留缓冲区大小的数据
#         process(example)

#     # 读取 sample-10BT 的全部 Parquet 文件
#     reader = ParquetReader(
#         "hf://datasets/HuggingFaceFW/fineweb/sample-10BT/data",
#         limit=None  # 不限数量，读全部
#     )
#     for doc in reader():
#         # doc 包含 'text', 'id', 'language_score' 等字段
#         process(doc)

#     main()


