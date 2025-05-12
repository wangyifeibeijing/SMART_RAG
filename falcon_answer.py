import os
import json
from concurrent.futures import ThreadPoolExecutor
from ai71 import AI71
from dotenv import load_dotenv
from chatclient import FalconChatClient
import pandas as pd
import jsonschema
from jsonschema import validate
    
# from langchain.prompts import PromptTemplate

# def build_prompt(question: str, passages: list) -> str:
#     """
#     Constructs a prompt for the Falcon model, instructing it to answer in the language of the input question.

#     :param question: The user's question.
#     :param passages: List of retrieved passages, each as a dictionary with 'passage' and 'doc_IDs'.
#     :return: A formatted prompt string.
#     """
#     # Format passages with their corresponding document IDs
#     passages_text = "\n".join(
#         [f"Passage {i+1}: {p['passage']}\nDocument IDs: {', '.join(p['doc_IDs'])}" for i, p in enumerate(passages)]
#     )

#     # Define the prompt template
#     template = (
#         "You are a helpful assistant. Below are some passages related to the user's question:\n\n"
#         "{passages}\n\n"
#         "The user's question is: {question}\n\n"
#         "Please answer the question in the same language as the question."
#     )

#     # Create and format the prompt
#     prompt = PromptTemplate.from_template(template)
#     return prompt.format(passages=passages_text, question=question)

# def rag_ver():
#     load_dotenv()
#     api_key = os.getenv("AI71_API_KEY")   
    

#     client = FalconChatClient(api_key = api_key)

#     def process_question(question_obj):
#         question = question_obj['question']
#         # 检索相关文档
#         passages = retrieve_documents(question)
#         # 构建提示词
#         prompt = build_prompt(question, passages)
#         # 调用 Falcon 模型
#         response = client.single_turn(prompt)
#         answer = response
#         # 构建结果对象
#         result = {
#             "id": question_obj['id'],
#             "question": question,
#             "passages": passages,
#             "final_prompt": prompt,
#             "answer": answer
#         }
#         return result

#     # 读取问题文件
#     with open("LiveRAG_LCD_Session1_Question_file.jsonl", "r") as f:
#         questions = [json.loads(line) for line in f]

#     # 并行处理问题
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         results = list(executor.map(process_question, questions))

#     # 保存结果到 answer.json
#     with open("answer.json", "w") as f:
#         json.dump(results, f, indent=2)

def falcon_only():
    from tqdm import tqdm  # 导入 tqdm 库

    # 加载环境变量
    load_dotenv()
    api_key = os.getenv("AI71_API_KEY")   
    client = FalconChatClient(api_key = api_key)

    # 读取问题文件
    with open("LiveRAG_LCD_Session1_Question_file.jsonl", "r") as f:
        questions = [json.loads(line) for line in f]

    def process_question(question):
        # 使用 retriever 获取相关段落
        passages = question["question"]

        # 构建 final_prompt
        final_prompt = f"Question: {question['question']}\nAnswer:"

        # 调用 Falcon 模型
        response = client.single_turn(final_prompt)
        answer = response
        passages = {
            "passage":"",
             "doc_IDs":[]
        }
        return {
            "id": question["id"],
            "question": question["question"],
            "passages": [passages],
            "final_prompt": final_prompt,
            "answer": answer
        }

    # 使用 tqdm 结合 executor 来显示进度条
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(executor.map(process_question, questions), total=len(questions), desc="Processing Questions"))




    

    # LiveRAG Answer JSON schema: 
    json_schema = """
    { 
    "$schema": "http://json-schema.org/draft-07/schema#", 

    "title": "Answer file schema", 
    "type": "object", 
    "properties": { 
        "id": { 
        "type": "integer", 
        "description": "Question ID" 
        }, 
        "question": { 
        "type": "string", 
        "description": "The question" 
        }, 
        "passages": { 
        "type": "array", 
        "description": "Passages used and related FineWeb doc IDs, ordered by decreasing importance", 
        "items": { 
            "type": "object", 
            "properties": {
            "passage": { 
                "type": "string", 
                "description": "Passage text" 
            }, 
            "doc_IDs": {
                "type": "array", 
                "description": "Passage related FineWeb doc IDs, ordered by decreasing importance", 
                "items": { 
                "type": "string", 
                "description": "FineWeb doc ID, e.g., <urn:uuid:d69cbebc-133a-4ebe-9378-68235ec9f091>"
                } 
            } 
            },
            "required": ["passage", "doc_IDs"]
        }
        }, 
        "final_prompt": {
        "type": "string",
        "description": "Final prompt, as submitted to Falcon LLM"
        },
        "answer": {
        "type": "string",
        "description": "Your answer"
        }
    },
    "required": ["id", "question", "passages", "final_prompt", "answer"]
    }
    """

    # Code that generates the output
    answers = pd.DataFrame(results)

    # Convert to JSON format
    answers_json = answers.to_json(orient='records', lines=True, force_ascii=False)

    # Or just save to a file
    answers.to_json("answers.jsonl", orient='records', lines=True, force_ascii=False)

    # Load the file to make sure it is ok
    loaded_answers = pd.read_json("answers.jsonl", lines=True)

    # Load the JSON schema
    schema = json.loads(json_schema)

    # Validate each Answer JSON object against the schema
    for answer in loaded_answers.to_dict(orient='records'):
        try:
            validate(instance=answer, schema=schema)
            print(f"Answer {answer['id']} is valid.")
        except jsonschema.exceptions.ValidationError as e:
            print(f"Answer {answer['id']} is invalid: {e.message}")








    # # 构造完整的 JSON 数据结构
    # output_data = {
    #     "$schema": "http://json-schema.org/draft-07/schema#",
    #     "title": "Answer file schema",
    #     "type": "array",
    #     "items": {
    #         "type": "object",
    #         "properties": {
    #             "id": {"type": "integer", "description": "Question ID"},
    #             "question": {"type": "string", "description": "The question"},
    #             "passages": {
    #                 "type": "array",
    #                 "description": "Passages used and related FineWeb doc IDs, ordered by decreasing importance",
    #                 "items": {
    #                     "type": "object",
    #                     "properties": {
    #                         "passage": {"type": "string", "description": "Passage text"},
    #                         "doc_IDs": {
    #                             "type": "array",
    #                             "description": "Passage related FineWeb doc IDs, ordered by decreasing importance",
    #                             "items": {"type": "string", "description": "FineWeb doc ID"}
    #                         }
    #                     },
    #                     "required": ["passage", "doc_IDs"]
    #                 }
    #             },
    #             "final_prompt": {"type": "string", "description": "Final prompt, as submitted to Falcon LLM"},
    #             "answer": {"type": "string", "description": "Your answer"}
    #         },
    #         "required": ["id", "question", "passages", "final_prompt", "answer"]
    #     },
    #     "data": results  # 包含所有问题的回答数据
    # }

    # # 将 JSON 数据写入文件
    # with open("answers.json", "w", encoding="utf-8") as f:
    #     json.dump(output_data, f, ensure_ascii=False, indent=4)

    # # 保存结果
    # with open("answer.json", "w") as f:
    #     json.dump(results, f, indent=4)

    # print("Processing complete. Results saved to answer.json.")
    # load_dotenv()
    # api_key = os.getenv("AI71_API_KEY")   
    

    # client = FalconChatClient(api_key = api_key)

    # def process_question(question_obj):
    #     question = question_obj['question']
        
    #     # 构建提示词
    #     prompt = question
    #     # 调用 Falcon 模型
    #     response = client.single_turn(prompt)
    #     answer = response
    #     # 构建结果对象
    #     result = {
    #         "id": question_obj['id'],
    #         "question": question,
    #         "passages": None,
    #         "final_prompt": prompt,
    #         "answer": answer
    #     }
    #     return result

    # # 读取问题文件
    # with open("LiveRAG_LCD_Session1_Question_file.jsonl", "r") as f:
    #     questions = [json.loads(line) for line in f]

    # # 并行处理问题
    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     results = list(executor.map(process_question, questions))

    # # 保存结果到 answer.json
    # with open("answer.json", "w") as f:
    #     json.dump(results, f, indent=2)






if __name__ == "__main__":
    falcon_only()