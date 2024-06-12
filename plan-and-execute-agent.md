# Plan and execute agent

여기서는 Plan and execute를 하는 Agent를 구성합니다. 상세한 내용은 [planning-agents.md](https://github.com/kyopark2014/llm-agent/blob/main/planning-agents.md)을 참조합니다.


## 요약

- Cycle을 이용해 구성한 Workflow는 복잡한 문제를 해결할 수 있도록 해줍니다.
- 사용하는 API로 해결이 안되는 Plan이 생성되면 루프를 돌다가 결과를 얻지 못하고 실패합니다. 어떤 API를 어떤식으로 사용할지 추가 study가 필요합니다.

## 구성 방법

Plan and execute의 구성도는 아래와 같습니다. 

![image](https://github.com/kyopark2014/adoptive-agent/assets/52392004/084cc0b6-6374-44b7-a63c-1af4494b63f4)

아래와 같이 Plan을 Prompt를 이용해 구현합니다. pydantic을 이용하기 위해 ChatBedrock이 아닌 AnthropicBedrock과 instructor.from_anthropic()을 이용하였습니다.

```python
class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str]

def generate_plans(text):
    client = instructor.from_anthropic(
        AnthropicBedrock(
            aws_region="us-west-2",
        )
    )
    
    if(mode=='eng'):
        system = """For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps."""
    else:
        system = """주어진 목표에 대해 간단한 단계별 계획을 세웁니다. 이 계획에는 개별 작업이 포함되어 있으며, 이를 올바르게 실행하면 정확한 답을 얻을 수 있습니다. \
    불필요한 단계는 추가하지 마십시오. 마지막 단계의 결과가 최종 답이 되어야 합니다. 각 단계에 필요한 모든 정보가 포함되어 있는지 확인하고 단계를 건너뛰지 마십시오."""
    
    resp = client.messages.create(
        model="anthropic.claude-3-sonnet-20240229-v1:0", # model="anthropic.claude-3-haiku-20240307-v1:0"
        max_tokens=1024,
        system = system,
        messages=[
            {"role": "user","content": text}
        ],
        response_model=Plan,
    )    
    
    return resp.steps
```

PlanExecute class를 지정하고 plan_step을 정의합니다. generate_plans()으로 얻은 plan들을 기반으로 input을 재정의합니다. 

```python
class PlanExecute(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[List[Tuple], operator.add]
    
    response: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    
def plan_step(state: PlanExecute):
    print('state: ', state)
    
    plan = generate_plans(state['input'])
    print('plan: ', plan)
    
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"For the following plan: {plan_str}\n\nYou are tasked with executing step {1}, {task}."    
    print('task_formatted: ', task_formatted)
    
    return {
        "input": task_formatted,
        "plan": plan
    }
```

Single Task를 구행하는 ReAct Agent를 create_react_agent()을 이용해 생성합니다. run_agent_plan()에서는 PlanExecute Class형태로 데이터를 받아서, Agent로 실행합니다. Single task를 수행하기 위해 Agent안에서 thought-action-observation의 동작을 수행합니다.

```python
prompt_template = get_react_prompt_template(agentLangMode)
agent_plan = create_react_agent(chat, tools, prompt_template)

def run_agent_plan(state: PlanExecute):
    print('state: ', state)
    
    agent_outcome = agent_plan.invoke(state)
    print('agent_outcome: ', agent_outcome)
    
    return {"agent_outcome": agent_outcome}
```

Replan은 아래와 같이 정의합니다. pydantic을 쓰고자 했으나 parsing을 못하는 문제가 있어서 prompt를 이용하였습니다. 

```python
class Response(BaseModel):
    """Response to user."""

    response: str

class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan]

def replan_step(state: PlanExecute):
    print('state: ', state)
    
    input = state['input']
    plan = state["plan"]    
    past_steps = state['past_steps']
    
    #client = instructor.from_anthropic(
    #    AnthropicBedrock(
    #        aws_region="us-west-2",
    #    )
    #)
    client = AnthropicBedrock(
            aws_region="us-west-2",
        )

    message = f"""For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.   
The updated plan should be in the following format:
<plan>
[\"<step>\", \"<step>\", ...]
</plan>
"""
    # print('message: ', message)
    
    output = client.messages.create(
        model="anthropic.claude-3-sonnet-20240229-v1:0", # model="anthropic.claude-3-haiku-20240307-v1:0"
        max_tokens=1024,
        messages=[
            {"role": "user","content": message}
        ],
        # response_model=Act,
    )    
    print('output: ', output)

    result = output.content
    
    value = (result[0].text).replace("\n","")
    
    plan = json.loads(value[value.find('<plan>')+6:value.find('</plan>')])
    
    if isinstance(state["agent_outcome"], AgentFinish):
        return {"response": value}
    else:
        return {"plan": plan}
```

새로운 plan은 아래와 같이 문장으로 제공됩니다.

```text
Since the 2024 Australian Open tennis tournament has not happened yet, 
the plan should be updated as follows:\n\nPlan:\n
['If no information is available on the 2024 winner yet, respond that the tournament has not happened yet']\n\n
As the final step, I will respond that the tournament has not happened yet, since no information is available on the 2024 winner.
```

따라서, Prompt를 이용해 <plan></plan> 테그를 붙이고 포맷을 list 형태로 지정하였습니다. 프롬프트 이후로 생성된 결과는 아래와 같습니다. 

```python
<plan>\n["1. Check the current date and compare it to when the 2024 Australian Open is scheduled to take place (around mid-to-late January)"]\n</plan>
```

여기서 list만 꺼내기 위해 <plan></plan> tag를 없애고 "\n"을 제거한후 json.load()를 이용해 변환합니다.

Graph에 대한 Workshop를 아래와 같이 지정합니다.

```python
def buildAgent():
    workflow = StateGraph(PlanExecute)
    
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", run_agent_plan)
    workflow.add_node("replan", replan_step)
    
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        should_continue,
        {
            "continue": "agent",
            "end": END,
        },
    )
    
    return workflow.compile()
app_plan = buildAgent()

def run_plan_and_execute(connectionId, requestId, app, query):
    isTyping(connectionId, requestId)
    
    inputs = {"input": query}
    
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "passenger_id": "3442 587242",
            "thread_id": thread_id,
        },
        "recursion_limit": 50
    }
    for output in app.stream(inputs, config=config):
        for key, value in output.items():
            print("---")
            print(f"Node '{key}': {value}")
            
            if 'agent_outcome' in value and isinstance(value['agent_outcome'], AgentFinish):
                response = value['agent_outcome'].return_values
                msg = readStreamMsg(connectionId, requestId, response['output'])

    return msg
```
