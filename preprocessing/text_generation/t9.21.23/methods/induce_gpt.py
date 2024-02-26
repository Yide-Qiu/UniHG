
import openai
import pdb
import os
openai.api_key = "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"
# pdb.set_trace()

origin_txt = "attribute-all.txt"
# proxy_url = 'cdn-cn.nekocloud.cn:19016'
# os.environ['HTTP_PROXY'] = proxy_url
# os.environ['HTTPS_PROXY'] = proxy_url

with open(origin_txt, 'r', encoding='utf-8') as file:
  for line in file:
    text_input = line 
    prompt = "To summarize the main idea of this paragraph, it should involve various aspects mentioned."
    text_prompt = f"""
    {prompt}:\n
    '''{text_input}'''
    
    """

    # 调用GPT-3.5模型的API
    response = openai.Completion.create(
      # engine="text-davinci-003",  # 指定使用的GPT-3.5模型引擎
      
      engine="text-embedding-ada-002"
      prompt=text_prompt,  # 输入文本
      max_tokens=100,  # 生成的文本长度
      n=1,  # 生成的候选回复数目
      stop=None,  # 可选参数，指定生成的终止词
      temperature=0.8,  # 可选参数，用于控制生成文本的多样性，值越大越随机
    )

    # 提取生成的文本
    pdb.set_trace()
    embedding = response.choices[0].embedding
    generated_text = response.choices[0].text.strip()
    print(embedding)
    print(generated_text)




















