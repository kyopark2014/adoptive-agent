# Structured Output Issue

## Issue: Structured output 미지원

[chat models](https://python.langchain.com/v0.2/docs/integrations/chat/)와 같이 ChatBedrock의 경우에 Structured output을 미지원하고 있습니다. [관련한 ticket](https://github.com/langchain-ai/langchain/discussions/22701)이 있어서 모니터링 중입니다. 

<img src="https://github.com/kyopark2014/adoptive-agent/assets/52392004/cac50362-93a8-40a3-a516-69aecc4f3611" width="500">

따라서, with_structured_output을 사용할 수 없습니다.

## 해결을 위한 참고자료

[Unlocking Structured Outputs with Amazon Bedrock: A Guide to Leveraging Instructor and Anthropic Claude 3](https://medium.com/@dminhk/unlocking-structured-outputs-with-amazon-bedrock-a-guide-to-leveraging-instructor-and-anthropic-abb76e4f6b20)와 [Langchain workaround for with_structured_output using ChatBedrock](https://stackoverflow.com/questions/78472764/langchain-workaround-for-with-structured-output-using-chatbedrock)을 참조합니다. 

```python
from typing import List
import instructor
from anthropic import AnthropicBedrock
from loguru import logger
from pydantic import BaseModel
import enum

class User(BaseModel):
    name: str
    age: int

class MultiLabels(str, enum.Enum):
    TECH_ISSUE = "tech_issue"
    BILLING = "billing"
    GENERAL_QUERY = "general_query"

class MultiClassPrediction(BaseModel):
    """
    Class for a multi-class label prediction.
    """
    class_labels: List[MultiLabels]

if __name__ == "__main__":
    # Initialize the instructor client with AnthropicBedrock configuration
    client = instructor.from_anthropic(
        AnthropicBedrock(
            aws_region="eu-central-1",
        )
    )

    logger.info("Hello World Example")

    # Create a message and extract user data
    resp = client.messages.create(
        model="anthropic.claude-instant-v1",
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
    logger.info("Classification Example")

    # Classify a support ticket
    text = "My account is locked and I can't access my billing info."

    _class = client.chat.completions.create(
        model="anthropic.claude-instant-v1",
        max_tokens=1024,
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following support ticket: {text}",
            },
        ],
    )

    print(_class)
```

필요한 패키지는 아래와 같습니다. 

```python
pip install -qU instructor
```
