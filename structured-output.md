# Structured Output Issue (해결됨)

## Updae: structured output 지원 

[관련 Ticket](https://github.com/langchain-ai/langchain/pull/23645)

![image](https://github.com/user-attachments/assets/28ee7e86-e6cf-45e5-b47f-9da380d2ddce)

## 사용법

```python
from botocore.config import Config
from langchain_aws import ChatBedrock
bedrock_region = 'us-east-1'
modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }            
    )
)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"
maxOutputTokens = 4096
parameters = {
    "max_tokens":maxOutputTokens,     
    "temperature":0.1,
    "top_k":250,
    "top_p":0.9,
    "stop_sequences": [HUMAN_PROMPT]
}    
chat = ChatBedrock(   
    model_id=modelId,
    client=boto3_bedrock, 
    model_kwargs=parameters,
)

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''
    answer: str
    justification: str
    
structured_llm = chat.with_structured_output(AnswerWithJustification, include_raw=True)

structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
```

이때의 결과는 아래와 같습니다. 

```java
{
   "raw":"AIMessage(content=""",
   "additional_kwargs="{
      "usage":{
         "prompt_tokens":361,
         "completion_tokens":156,
         "total_tokens":517
      },
      "stop_reason":"tool_use",
      "model_id":"anthropic.claude-3-sonnet-20240229-v1:0"
   },
   "response_metadata="{
      "usage":{
         "prompt_tokens":361,
         "completion_tokens":156,
         "total_tokens":517
      },
      "stop_reason":"tool_use",
      "model_id":"anthropic.claude-3-sonnet-20240229-v1:0"
   },
   "id=""run-d8002024-600e-4962-a763-af6e785d87ed-0",
   "tool_calls="[
      {
         "name":"AnswerWithJustification",
         "args":{
            "answer":"A pound of bricks and a pound of feathers weigh the same.",
            "justification":"A pound is a unit of weight or mass, not volume. Since a pound of bricks and a pound of feathers both have the same mass (one pound), they must weigh the same amount. The fact that bricks are denser and take up less volume than feathers for the same weight is irrelevant - their weights are equal when the mass is the same. This is a classic example that illustrates the difference between weight and density."
         },
         "id":"toolu_bdrk_019AZDSDKrTHhJRBKLpkjmTU"
      }
   ],
   "usage_metadata="{
      "input_tokens":361,
      "output_tokens":156,
      "total_tokens":517
   }")",
   "parsed":"AnswerWithJustification(answer=""A pound of bricks and a pound of feathers weigh the same.",
   "justification=""A pound is a unit of weight or mass, not volume. Since a pound of bricks and a pound of feathers both have the same mass (one pound), they must weigh the same amount. The fact that bricks are denser and take up less volume than feathers for the same weight is irrelevant - their weights are equal when the mass is the same. This is a classic example that illustrates the difference between weight and density."")",
   "parsing_error":"None"
}
```

# Previous 우회 방법

ChatBedrock의 경우에 현재 Structured Output을 지원하지 않고 있습니다. 이를 Prompt를 이용해 우회했는데, 복잡할 뿐 아니라 결과에도 영향을 주어서, Structured Output을 구현한 블로그를 참조하여 우회하였습니다.

## Issue: Structured output 미지원

[chat models](https://python.langchain.com/v0.2/docs/integrations/chat/)와 같이 ChatBedrock의 경우에 Structured output을 미지원하고 있습니다. [관련한 ticket](https://github.com/langchain-ai/langchain/discussions/22701)도 있지만 현재 진행이되고 있지 않습니다. 따라서, with_structured_output을 현재 사용할 수 없습니다. (2024.6.12 기준)

<img src="https://github.com/kyopark2014/adoptive-agent/assets/52392004/cac50362-93a8-40a3-a516-69aecc4f3611" width="700">

## Anthropic Bedrock

AnthropicBedrock은 [Amazon Bedrock API](https://docs.anthropic.com/en/api/claude-on-amazon-bedrock)을 이용하며, [Unlocking Structured Outputs with Amazon Bedrock: A Guide to Leveraging Instructor and Anthropic Claude 3](https://medium.com/@dminhk/unlocking-structured-outputs-with-amazon-bedrock-a-guide-to-leveraging-instructor-and-anthropic-abb76e4f6b20)을 참조하면 Bedrock에서 Structured output을 사용할 수 있습니다. 추가적인 코드는 [Langchain workaround for with_structured_output using ChatBedrock](https://stackoverflow.com/questions/78472764/langchain-workaround-for-with-structured-output-using-chatbedrock)을 참조합니다. 

### 구현 결과

아래와 같이 name, age를 User에 넣고, "Jason is 25 years old."에 대해 추출할 수 있습니다.

```python
import instructor
from anthropic import AnthropicBedrock
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

def extract_user_info(text):    
    client = instructor.from_anthropic(
        AnthropicBedrock(
            aws_region="us-west-2",
        )
    )
    
    resp = client.messages.create(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=User,
    )

    print(resp)

extract_user_info("Jason is 25 years old.")
```

이때의 결과는 아래와 같습니다.

```text
name='Jason' age=25
```

