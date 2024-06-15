import json
import boto3
import os
import time
import datetime
import PyPDF2
import csv
import re
import traceback
import requests
import base64
import operator
import uuid

from botocore.config import Config
from botocore.exceptions import ClientError
from io import BytesIO
from urllib import parse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_aws import ChatBedrock
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.agents import tool
from langchain.agents import AgentExecutor, create_react_agent
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pytz import timezone
from langchain_community.tools.tavily_search import TavilySearchResults
from PIL import Image
from opensearchpy import OpenSearch

from typing import List, Tuple, Annotated, TypedDict, Literal, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import END, StateGraph
from langchain_core.runnables import ensure_config
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
path = os.environ.get('path')
doc_prefix = s3_prefix+'/'
debugMessageMode = os.environ.get('debugMessageMode', 'false')
agentLangMode = 'kor'
projectName = os.environ.get('projectName')
opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
opensearch_url = os.environ.get('opensearch_url')
LLM_for_chat = json.loads(os.environ.get('LLM_for_chat'))
LLM_for_multimodal= json.loads(os.environ.get('LLM_for_multimodal'))
LLM_embedding = json.loads(os.environ.get('LLM_embedding'))
selected_chat = 0
selected_multimodal = 0
selected_embedding = 0
separated_chat_history = os.environ.get('separated_chat_history')
enalbeParentDocumentRetrival = os.environ.get('enalbeParentDocumentRetrival')

# api key to get weather information in agent
secretsmanager = boto3.client('secretsmanager')
try:
    get_weather_api_secret = secretsmanager.get_secret_value(
        SecretId=f"openweathermap-{projectName}"
    )
    #print('get_weather_api_secret: ', get_weather_api_secret)
    secret = json.loads(get_weather_api_secret['SecretString'])
    #print('secret: ', secret)
    weather_api_key = secret['weather_api_key']

except Exception as e:
    raise e
   
# api key to use LangSmith
langsmith_api_key = ""
try:
    get_langsmith_api_secret = secretsmanager.get_secret_value(
        SecretId=f"langsmithapikey-{projectName}"
    )
    #print('get_langsmith_api_secret: ', get_langsmith_api_secret)
    secret = json.loads(get_langsmith_api_secret['SecretString'])
    #print('secret: ', secret)
    langsmith_api_key = secret['langsmith_api_key']
    langchain_project = secret['langchain_project']
except Exception as e:
    raise e

if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

# api key to use Tavily Search
tavily_api_key = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    #print('secret: ', secret)
    tavily_api_key = secret['tavily_api_key']
except Exception as e: 
    raise e

if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key
   
# websocket
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi', endpoint_url=connection_url)
print('connection_url: ', connection_url)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

map_chain = dict() 
map_task = dict() 
map_app = dict()

MSG_LENGTH = 100

# Multi-LLM
def get_chat():
    global selected_chat
    
    profile = LLM_for_chat[selected_chat]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'selected_chat: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    maxOutputTokens = int(profile['maxOutputTokens'])
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )
    
    selected_chat = selected_chat + 1
    if selected_chat == len(LLM_for_chat):
        selected_chat = 0
    
    return chat

def get_multimodal():
    profile = LLM_for_multimodal[selected_multimodal]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'selected_multimodal: {selected_multimodal}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    maxOutputTokens = int(profile['maxOutputTokens'])
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    multimodal = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    selected_multimodal = selected_multimodal + 1
    if selected_multimodal == len(LLM_for_multimodal):
        selected_multimodal = 0
    
    return multimodal

def get_embedding():
    global selected_embedding
    profile = LLM_embedding[selected_embedding]
    bedrock_region =  profile['bedrock_region']
    model_id = profile['model_id']
    print(f'selected_embedding: {selected_embedding}, bedrock_region: {bedrock_region}, model_id:{model_id}')
    
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region, 
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    
    bedrock_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name = bedrock_region,
        model_id = model_id
    )  
    
    selected_embedding = selected_embedding + 1
    if selected_embedding == len(LLM_embedding):
        selected_embedding = 0
    
    return bedrock_embedding

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
    
    return texts

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(chat, docs):    
    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        print('result of summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary
    
def load_chatHistory(userId, allowTime, chat_memory):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            print('text: ', text)
            print('msg: ', msg)        

            chat_memory.save_context({"input": text}, {"output": msg})             

def getAllowTime():
    d = datetime.datetime.now() - datetime.timedelta(days = 2)
    timeStr = str(d)[0:19]
    print('allow time: ',timeStr)

    return timeStr

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False

def general_conversation(connectionId, requestId, chat, query):
    if isKorean(query)==True :
        system = (
            "다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor."
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
        
        usage = stream.response_metadata['usage']
        print('prompt_tokens: ', usage['prompt_tokens'])
        print('completion_tokens: ', usage['completion_tokens'])
        print('total_tokens: ', usage['total_tokens'])
        msg = stream.content

    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def get_documents_from_opensearch(vectorstore_opensearch, query, top_k):
    result = vectorstore_opensearch.similarity_search_with_score(
        query = query,
        k = top_k*2,  
        pre_filter={"doc_level": {"$eq": "child"}}
    )
    print('result: ', result)
            
    relevant_documents = []
    docList = []
    for re in result:
        if 'parent_doc_id' in re[0].metadata:
            parent_doc_id = re[0].metadata['parent_doc_id']
            doc_level = re[0].metadata['doc_level']
            print(f"doc_level: {doc_level}, parent_doc_id: {parent_doc_id}")
                    
            if doc_level == 'child':
                if parent_doc_id in docList:
                    print('duplicated!')
                else:
                    relevant_documents.append(re)
                    docList.append(parent_doc_id)
                    
                    if len(relevant_documents)>=top_k:
                        break
                                
    # print('lexical query result: ', json.dumps(response))
    print('relevant_documents: ', relevant_documents)
    
    return relevant_documents

os_client = OpenSearch(
    hosts = [{
        'host': opensearch_url.replace("https://", ""), 
        'port': 443
    }],
    http_compress = True,
    http_auth=(opensearch_account, opensearch_passwd),
    use_ssl = True,
    verify_certs = True,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
)

def get_parent_document(parent_doc_id):
    response = os_client.get(
        index="idx-rag", 
        id = parent_doc_id
    )
    
    source = response['_source']                            
    # print('parent_doc: ', source['text'])   
    
    metadata = source['metadata']    
    #print('name: ', metadata['name'])   
    #print('uri: ', metadata['uri'])   
    #print('doc_level: ', metadata['doc_level']) 
    
    return source['text'], metadata['name'], metadata['uri'], metadata['doc_level']    

@tool 
def get_book_list(keyword: str) -> str:
    """
    Search book list by keyword and then return book list
    keyword: search keyword
    return: book list
    """
    
    keyword = keyword.replace('\'','')

    #answer = ""
    output = []   
    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        prod_info = soup.find_all("a", attrs={"class": "prod_info"})
        
        #if len(prod_info):
        #    answer = "추천 도서는 아래와 같습니다.\n"
         
        for prod in prod_info[:5]:
            title = prod.text.strip().replace("\n", "")       
            link = prod.get("href")
            # answer = answer + f"{title}, URL: {link}\n"
            
            result = f"{title}, URL: {link}\n"
            output.append(result)
    
    return output
    
@tool
def get_current_time(format: str=f"%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in the specified format"""
    # f"%Y-%m-%d %H:%M:%S"
    format = format.replace('\'','')
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    # print('timestr:', timestr)
    
    return timestr

def get_lambda_client(region):
    return boto3.client(
        service_name='lambda',
        region_name=region
    )

@tool    
def get_system_time() -> list:
    """
    retrive system time to earn the current date and time.
    return: a string of date and time
    """    
    
    function_name = "lambda-datetime-for-llm-agent"
    lambda_region = 'ap-northeast-2'
    
    try:
        lambda_client = get_lambda_client(region=lambda_region)
        payload = {}
        print("Payload: ", payload)
            
        response = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps(payload),
        )
        print("Invoked function %s.", function_name)
        print("Response: ", response)
    except ClientError:
        print("Couldn't invoke function %s.", function_name)
        raise
    
    payload = response['Payload']
    print('payload: ', payload)
    body = json.load(payload)['body']
    print('body: ', body)
    jsonBody = json.loads(body) 
    print('jsonBody: ', jsonBody)    
    timestr = jsonBody['timestr']
    print('timestr: ', timestr)
    
    result = []
    result.append(timestr)
    
    return result

@tool
def get_weather_info(city: str) -> str:
    """
    retrieve weather information by city name and then return weather statement.
    city: the name of city to retrieve
    return: weather statement
    """    
    
    city = city.replace('\n','')
    city = city.replace('\'','')
    city = city.replace('\"','')
                
    chat = get_chat()
    if isKorean(city):
        place = traslation(chat, city, "Korean", "English")
        print('city (translated): ', place)
    else:
        place = city
        city = traslation(chat, city, "English", "Korean")
        print('city (translated): ', city)
        
    print('place: ', place)
    
    weather_str: str = f"{city}에 대한 날씨 정보가 없습니다."
    if weather_api_key: 
        apiKey = weather_api_key
        lang = 'en' 
        units = 'metric' 
        api = f"https://api.openweathermap.org/data/2.5/weather?q={place}&APPID={apiKey}&lang={lang}&units={units}"
        # print('api: ', api)
                
        try:
            result = requests.get(api)
            result = json.loads(result.text)
            print('result: ', result)
        
            if 'weather' in result:
                overall = result['weather'][0]['main']
                current_temp = result['main']['temp']
                min_temp = result['main']['temp_min']
                max_temp = result['main']['temp_max']
                humidity = result['main']['humidity']
                wind_speed = result['wind']['speed']
                cloud = result['clouds']['all']
                
                weather_str = f"{city}의 현재 날씨의 특징은 {overall}이며, 현재 온도는 {current_temp}도 이고, 최저온도는 {min_temp}도, 최고 온도는 {max_temp}도 입니다. 현재 습도는 {humidity}% 이고, 바람은 초당 {wind_speed} 미터 입니다. 구름은 {cloud}% 입니다."
                #weather_str = f"Today, the overall of {city} is {overall}, current temperature is {current_temp} degree, min temperature is {min_temp} degree, highest temperature is {max_temp} degree. huminity is {humidity}%, wind status is {wind_speed} meter per second. the amount of cloud is {cloud}%."            
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                    
            # raise Exception ("Not able to request to LLM")    
        
    print('weather_str: ', weather_str)                            
    return weather_str

@tool
def search_by_tavily(keyword: str) -> str:
    """
    Search general information by keyword and then return the result as a string.
    keyword: search keyword
    return: the information of keyword
    """    
    
    answer = ""
    
    if tavily_api_key:
        keyword = keyword.replace('\'','')
        
        search = TavilySearchResults(k=3)
                    
        output = search.invoke(keyword)
        print('tavily output: ', output)
        
        for result in output:
            print('result: ', result)
            if result:
                content = result.get("content")
                url = result.get("url")
            
                answer = answer + f"{content}, URL: {url}\n"
        
    return answer

@tool    
def search_by_opensearch(keyword: str) -> str:
    """
    Search technical information by keyword and then return the result as a string.
    keyword: search keyword
    return: the technical information of keyword
    """    
    
    print('keyword: ', keyword)
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    print('modified keyword: ', keyword)
    
    bedrock_embedding = get_embedding()
        
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = "idx-*", # all
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    ) 
    
    answer = ""
    top_k = 2
    
    if enalbeParentDocumentRetrival == 'true':
        relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, keyword, top_k)

        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']
            print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
                
            excerpt, name, uri, doc_level = get_parent_document(parent_doc_id) # use pareant document
            print(f"parent: name: {name}, uri: {uri}, doc_level: {doc_level}")
            
            answer = answer + f"{excerpt}, URL: {uri}\n\n"
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = keyword,
            k = top_k,
        )

        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')
            
            excerpt = document[0].page_content        
            uri = document[0].metadata['uri']
                            
            answer = answer + f"{excerpt}, URL: {uri}\n\n"
    
    return answer

# define tools
tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_opensearch]        

def get_react_prompt_template(mode: str): # (hwchase17/react) https://smith.langchain.com/hub/hwchase17/react
    # Get the react prompt template
    
    if mode=='eng':
        return PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should use only the tool name from [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 5 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
    else: 
        return PromptTemplate.from_template("""다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

사용할 수 있는 tools은 아래와 같습니다:

{tools}

다음의 format을 사용하세요.:

Question: 답변하여야 할 input question 
Thought: you should always think about what to do. 
Action: 해야 할 action로서 [{tool_names}]에서 tool의 name만을 가져옵니다. 
Action Input: action의 input
Observation: action의 result
... (Thought/Action/Action Input/Observation을 5번 반복 할 수 있습니다.)
Thought: 나는 이제 Final Answer를 알고 있습니다. 
Final Answer: original input에 대한 Final Answer

너는 Human에게 해줄 응답이 있거나, Tool을 사용하지 않아도 되는 경우에, 다음 format을 사용하세요.:
'''
Thought: Tool을 사용해야 하나요? No
Final Answer: [your response here]
'''

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
             
def run_agent_react(connectionId, requestId, userId, chat, query):
    prompt_template = get_react_prompt_template(agentLangMode)
    print('prompt_template: ', prompt_template)
    
    #from langchain import hub
    #prompt_template = hub.pull("hwchase17/react")
    #print('prompt_template: ', prompt_template)
    
     # create agent
    isTyping(connectionId, requestId)
    
    agent = create_react_agent(chat, tools, prompt_template)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations = 5
    )
    
    # run agent
    response = agent_executor.invoke({"input": query})
    print('response: ', response)

    # streaming    
    msg = readStreamMsg(connectionId, requestId, response['output'])

    msg = response['output']
    print('msg: ', msg)
            
    return msg

def get_react_chat_prompt_template(mode: str):
    # Get the react prompt template
    if mode=='eng':
        return PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should use only the tool name from [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 5 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Thought:{agent_scratchpad}
""")
    else: 

        return PromptTemplate.from_template("""다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.

사용할 수 있는 tools은 아래와 같습니다:

{tools}

다음의 format을 사용하세요.:

Question: 답변하여야 할 input question 
Thought: you should always think about what to do. 
Action: 해야 할 action로서 [{tool_names}]에서 tool의 name만을 가져옵니다. 
Action Input: action의 input
Observation: action의 result
... (Thought/Action/Action Input/Observation을 5번 반복 할 수 있습니다.)
Thought: 나는 이제 Final Answer를 알고 있습니다. 
Final Answer: original input에 대한 Final Answer

너는 Human에게 해줄 응답이 있거나, Tool을 사용하지 않아도 되는 경우에, 다음 format을 사용하세요.:
'''
Thought: Tool을 사용해야 하나요? No
Final Answer: [your response here]
'''

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Thought:{agent_scratchpad}
""")
    
    
def getMemoryTask(userId):
    # create memory_task
    if userId in map_task:  
        print('memory_task exist. reuse it!')        
        memory_task = map_task[userId]
    else: 
        print('memory_task does not exist. create new one!')                
        #memory_task = SqliteSaver.from_conn_string(":memory:")
        memory_task = AsyncSqliteSaver.from_conn_string(":memory:")
        map_task[userId] = memory_task    
        
####################### LangGraph #######################
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    user_id: str
    tool_output: json

chat = get_chat() 
mode  = 'kor'
prompt_template = get_react_prompt_template(mode)

def run_agent(state: AgentState):
    print('state: ', state)
    
    if not state['user_id']:
        config = ensure_config()  # update user_id
        configuration = config.get("configurable", {})
        # print('configuration: ', configuration)    
        user_id = configuration.get("user_id", None)
        print('user_id: ', user_id)    
        if not user_id:
            raise ValueError("No user_id configured.")
            
    agent_runnable = create_react_agent(chat, tools, prompt_template)
    
    agent_outcome = agent_runnable.invoke(state)
    
    return {
        "agent_outcome": agent_outcome,
        "user_id": user_id
    }
        
def execute_tools(state: AgentState):
    agent_action = state["agent_outcome"]
    # print(f"agent_action: {agent_action}")    
    print(f"tool: {agent_action.tool}")
    print(f"tool_input: {agent_action.tool_input}")
                
    #response = input(prompt=f"[y/n] continue with: {agent_action}?")
    #if response == "n":
    #    raise ValueError
        
    tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_opensearch]
    
    tool_executor = ToolExecutor(tools)
    
    #config = {
    #    "configurable": {
    #        "user_id": "3442 587242",
    #        # Checkpoints are accessed by thread_id
    #        "thread_id": "1234",
    #    }
    #}
    #output = tool_executor.invoke(agent_action, config)
    output = tool_executor.invoke(agent_action)
    print('output: ', output)
    
    if agent_action.tool == 'get_book_list':
        bookinfo = "추천 도서는 아래와 같습니다.\n"         
        for book in output:
            bookinfo = bookinfo + book + '\n'
        print('bookinfo: ', bookinfo)
        
        return {"intermediate_steps": [(agent_action, str(bookinfo))]}
    else:
        return {"intermediate_steps": [(agent_action, str(output))]}

def task_complete(state: AgentState):
    #print('state: ', state)
    
    if isinstance(state["agent_outcome"], AgentFinish):
        intermediate_steps = state["intermediate_steps"]
        for action, observation in intermediate_steps:
            #print(f"action: {action}")
            print(f"past task: {action.tool}")
            print(f"observation: {observation}")
            
            #msg = observation + "\n\n구매 하시겠어요?"
            
            #if action.tool == "get_book_list":            
            #    config = ensure_config()  # update userId
                
            #    current_state = app.get_state(config).values
            #    print('current_state: ', current_state)
                                    
            #response = input(prompt=f"[y/n] continue with: {observation}?")
            #if response == "n":
            #    raise ValueError
            
        return "end"
    else:
        return "continue"

def get_agent(userId):
    if userId in map_app:  
        print('memory_app exist. reuse it!')            
    else: 
        print('memory_app does not exist. create new one!')                
        memory_task = getMemoryTask(userId)
        
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", run_agent)
        workflow.add_node("action", execute_tools)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            task_complete,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")
        #app = workflow.compile(checkpointer=memory_task, interrupt_before=["action"])        
        app = workflow.compile(checkpointer=memory_task)
        map_app[userId] = app
        
    return map_app[userId]
    
def run_langgraph_agent(connectionId, requestId, userId, query):
    isTyping(connectionId, requestId)
        
    app = get_agent(userId)
    
    inputs = {"input": query}    
    config = {
        "configurable": {
            "thread_id": "thread-book",                
            "user_id": userId            
        },
        "recursion_limit": 50
    }
    for output in app.stream(inputs, config=config):
        print('output: ', output)
        for key, value in output.items():
            print("---")
            print(f"Node '{key}': {value}")
                        
            if 'agent_outcome' in value and isinstance(value['agent_outcome'], AgentFinish):
                response = value['agent_outcome'].return_values
                msg = readStreamMsg(connectionId, requestId, response['output'])
            
    return msg

####################### Bookstore bot #######################
class BookstoreState(TypedDict):
    input: str
    books: list[str]
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

def start_bookstore_agent(state: BookstoreState):
    print('state: ', state)
        
    config = ensure_config()  
    configuration = config.get("configurable", {})
    userId = configuration.get("user_id", None)
    print('userId: ', userId)
    if not userId:
        raise ValueError("No userId configured.")
    
    input = state['input']
    print('input: ', input)
    
    return {"input": input}
    # return AgentAction(tool=get_book_list, tool_input=state["input"])
    
def build_bookstore_agent():
    workflow = StateGraph(BookstoreState)

    workflow.add_node("entry", start_bookstore_agent)
    workflow.add_node("agent", run_agent)
    workflow.add_node("action", execute_tools)

    workflow.set_entry_point("entry")
    #workflow.set_entry_point("agent")
    workflow.add_edge("entry", "agent")
    workflow.add_conditional_edges(
        "agent",
        task_complete,
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")
    return workflow.compile()

app_bookstore = build_bookstore_agent()

def run_bookstore_bot(connectionId, requestId, userId, app, query):
    isTyping(connectionId, requestId)
    
    inputs = {"input": query}    

    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "user_id": userId,
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

    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "passenger_id": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }                                        
    return msg

####################### plan-and-execute agent #######################
query = "작년에 프로야구 우승팀이 누구지?"

from langchain_core.pydantic_v1 import Field

import instructor
from anthropic import AnthropicBedrock
from pydantic import BaseModel
from typing import Union

class Plan(BaseModel):
    """Plan to follow in future"""

    #steps: List[str] = Field(
    #    description="different steps to follow, should be in sorted order"
    #)
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
    # print("plan: ", resp.steps)
    
    return resp.steps

"""
def create_plan(chat, text):
    system = (
    "주어진 목표에 대해 간단한 단계별 계획을 세웁니다. 이 계획에는 개별 작업이 포함되어 있으며, 이를 올바르게 실행하면 정확한 답을 얻을 수 있습니다. \
    불필요한 단계는 추가하지 마십시오. 마지막 단계의 결과가 최종 답이 되어야 합니다. 각 단계에 필요한 모든 정보가 포함되어 있는지 확인하고 단계를 건너뛰지 마십시오. \
    결과만을 순서대로 아래 <example>과 같이 list로 정리하고, 번호는 붙이지 않습니다. 또한, 결과에 <result> tag를 붙여주세요. \
    <example>
    ["주요 언론사의 뉴스를 수집합니다.", "수집한 뉴스 기사들을 주제별로 분류합니다.", "각 주제별로 가장 많이 보도되고 화제가 된 뉴스를 선별합니다.", "최종적으로 선정된 소식을 정리합니다."]
    </example>
    ")
    human = "{input}"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat    
    
    result = chain.invoke({
        "input": text
    })
    output = result.content
    
    return output[output.find('<result>')+8:len(output)-9]

result = create_plan(chat, state['input'])
plan = json.loads(result.replace("\n",""))
print('plan: ', plan)
"""

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

prompt_template = get_react_prompt_template(agentLangMode)
agent_plan = create_react_agent(chat, tools, prompt_template)

def run_agent_plan(state: PlanExecute):
    print('state: ', state)
    
    agent_outcome = agent_plan.invoke(state)
    print('agent_outcome: ', agent_outcome)
    
    return {"agent_outcome": agent_outcome}
    #output = agent_response.content
    
    #return {
    #    "past_steps": (task, output),
    #}

class Response(BaseModel):
    """Response to user."""

    response: str

class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan]
    
    #action: Union[Response, Plan] = Field(
    #    description="Action to perform. If you want to respond to user, use Response. "
    #    "If you need to further use tools to get the answer, use Plan."
    #)
                
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
    
    #print("response: ", output.response)
    # Since the 2024 Australian Open tennis tournament has not happened yet, 
    # the plan should be updated as follows:\n\nPlan:\n
    # ['If no information is available on the 2024 winner yet, respond that the tournament has not happened yet']\n\n
    # As the final step, I will respond that the tournament has not happened yet, since no information is available on the 2024 winner.
    
    #<plan>\n[1. Check the current date and compare it to when the 2024 Australian Open is scheduled to take place (around mid-to-late January)]\n</plan>
    
    result = output.content
    print('result: ', result)
    
    value = (result[0].text).replace("\n","")
    print('value: ', value)
    
    plan = json.loads(value[value.find('<plan>')+6:value.find('</plan>')])
    print('plan: ', plan)
    
    if isinstance(state["agent_outcome"], AgentFinish):
        return {"response": value}
    else:
        return {"plan": plan}
        
    #return {
    #    "response": output.response
    #}
    
    #if isinstance(state["agent_outcome"], AgentFinish):
    #    return {"response": output.response}
    #else:
    #    return {"plan": state['plan']}
    
"""
def replan_step(state: PlanExecute):
    print('state: ', state)
    
    plan = state["plan"]    
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    print('plan_str: ', plan_str)
    
    task = plan[0]
    task_formatted = f"For the following plan: {plan_str}\n\nYou are tasked with executing step {1}, {task}."
    
    system = (
"For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan. \
결과만을 순서대로 아래 <example>과 같이 list로 정리하고, 번호는 붙이지 않습니다. 또한, 결과에 <result> tag를 붙여주세요. \
<example>
["주요 언론사의 뉴스를 수집합니다.", "수집한 뉴스 기사들을 주제별로 분류합니다.", "각 주제별로 가장 많이 보도되고 화제가 된 뉴스를 선별합니다.", "최종적으로 선정된 소식을 정리합니다."]
</example>    
"
)

    human = "{input}"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat    
    result = chain.invoke({
        "input": task_formatted,
        "plan": plan_str,
        "past_steps": state["past_steps"]
    })
    print('result: ', result)
    output = result.content
    print('output: ', output)
    
    result = output[output.find('<result>')+8:len(output)-9].replace("\n","")
    plan = json.loads(result)
    print('plan: ', plan)
    
    return {"plan": plan}
    
    #if isinstance(output.action, Response):
    #    return {"response": output.action.response}
    #else:
    #    return {"plan": output.action.steps}
"""

def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"
    
def buildAgent():
    workflow = StateGraph(PlanExecute)
    
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", run_agent_plan)
    workflow.add_node("replan", replan_step)
    
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")

    #workflow.add_conditional_edges(
    #    "replan",
    #    should_end,
    #)
    workflow.add_conditional_edges(
        "replan",
        task_complete,
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

def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def summary_of_code(chat, code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "code": code
            }
        )
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def revise_question(connectionId, requestId, chat, query):    
    global history_length, token_counter_history    
    history_length = token_counter_history = 0
        
    if isKorean(query)==True :      
        system = (
            ""
        )  
        human = """이전 대화를 참조하여, 다음의 <question>의 뜻을 명확히 하는 새로운 질문을 한국어로 생성하세요. 새로운 질문은 원래 질문의 중요한 단어를 반드시 포함합니다. 결과는 <result> tag를 붙여주세요.
        
        <question>            
        {question}
        </question>"""
        
    else: 
        system = (
            ""
        )
        human = """Rephrase the follow up <question> to be a standalone question. Put it in <result> tags.
        <question>            
        {question}
        </question>"""
            
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "history": history,
                "question": query,
            }
        )
        generated_question = result.content
        
        revised_question = generated_question[generated_question.find('<result>')+8:len(generated_question)-9] # remove <result> tag                   
        print('revised_question: ', revised_question)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    if debugMessageMode == 'true':  
        chat_history = ""
        for dialogue_turn in history:
            #print('type: ', dialogue_turn.type)
            #print('content: ', dialogue_turn.content)
            
            dialog = f"{dialogue_turn.type}: {dialogue_turn.content}\n"            
            chat_history = chat_history + dialog
                
        history_length = len(chat_history)
        print('chat_history length: ', history_length)
        
        token_counter_history = 0
        if chat_history:
            token_counter_history = chat.get_num_tokens(chat_history)
            print('token_size of history: ', token_counter_history)
            
        sendDebugMessage(connectionId, requestId, f"새로운 질문: {revised_question}\n * 대화이력({str(history_length)}자, {token_counter_history} Tokens)을 활용하였습니다.")
            
    return revised_question    
    # return revised_question.replace("\n"," ")

def isTyping(connectionId, requestId):    
    msg_proceeding = {
        'request_id': requestId,
        'msg': 'Proceeding...',
        'status': 'istyping'
    }
    #print('result: ', json.dumps(result))
    sendMessage(connectionId, msg_proceeding)

def removeFunctionXML(msg):
    #print('msg: ', msg)
    
    while(1):
        start_index = msg.find('<function_calls>')
        end_index = msg.find('</function_calls>')
        length = 18
        
        if start_index == -1:
            start_index = msg.find('<invoke>')
            end_index = msg.find('</invoke>')
            length = 10
        
        output = ""
        if start_index>=0:
            # print('start_index: ', start_index)
            # print('msg: ', msg)
            
            if start_index>=1:
                output = msg[:start_index-1]
                
                if output == "\n" or output == "\n\n":
                    output = ""
            
            if end_index >= 1:
                # print('end_index: ', end_index)
                output = output + msg[end_index+length:]
                            
            msg = output
        else:
            output = msg
            break

    return output

def readStreamMsgForAgent(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event
            
            output = removeFunctionXML(msg)
            # print('output: ', output)
            
            if len(output)>0 and output[0]!='<':
                result = {
                    'request_id': requestId,
                    'msg': output,
                    'status': 'proceeding'
                }
                #print('result: ', json.dumps(result))
                sendMessage(connectionId, result)
            
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event
            
            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            #print('result: ', json.dumps(result))
            sendMessage(connectionId, result)
    # print('msg: ', msg)
    return msg
    
def sendMessage(id, body):
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        raise Exception ("Not able to send a message")

def sendResultMessage(connectionId, requestId, msg):    
    result = {
        'request_id': requestId,
        'msg': msg,
        'status': 'completed'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, result)
    
def sendDebugMessage(connectionId, requestId, msg):
    debugMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'debug'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, debugMsg)    
        
def sendErrorMessage(connectionId, requestId, msg):
    errorMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'error'
    }
    print('error: ', json.dumps(errorMsg))
    sendMessage(connectionId, errorMsg)    

def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            memory_chain.chat_memory.add_user_message(text)
            if len(msg) > MSG_LENGTH:
                memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
            else:
                memory_chain.chat_memory.add_ai_message(msg)     

def translate_text(chat, text):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def check_grammer(chat, text):
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        msg = result.content
        print('result of grammer correction: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return msg

def use_multimodal(img_base64, query):    
    multimodal = get_multimodal()
    
    if query == "":
        query = "그림에 대해 상세히 설명해줘."
    
    messages = [
        SystemMessage(content="답변은 500자 이내의 한국어로 설명해주세요."),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = multimodal.invoke(messages)
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def extract_text(chat, img_base64):    
    query = "텍스트를 추출해서 utf8로 변환하세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        extracted_text = result.content
        print('result of text extraction from an image: ', extracted_text)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return extracted_text

def getResponse(connectionId, jsonBody):
    print('jsonBody: ', jsonBody)
    
    userId  = jsonBody['user_id']
    print('userId: ', userId)
    requestId  = jsonBody['request_id']
    print('requestId: ', requestId)
    requestTime  = jsonBody['request_time']
    print('requestTime: ', requestTime)
    type  = jsonBody['type']
    print('type: ', type)
    body = jsonBody['body']
    print('body: ', body)
    convType = jsonBody['convType']
    print('convType: ', convType)
    
    global map_chain, memory_chain, map_task
    
    # Multi-LLM
    profile = LLM_for_chat[selected_chat]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'selected_chat: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    # print('profile: ', profile)
    
    chat = get_chat()    
    
    # create memory_chain
    if userId in map_chain:  
        print('memory exist. reuse it!')        
        memory_chain = map_chain[userId]
    else: 
        print('memory_chain does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

        allowTime = getAllowTime()
        load_chat_history(userId, allowTime)
        
    start = int(time.time())    

    msg = ""
    if type == 'text' and body[:11] == 'list models':
        bedrock_client = boto3.client(
            service_name='bedrock',
            region_name=bedrock_region,
        )
        modelInfo = bedrock_client.list_foundation_models()    
        print('models: ', modelInfo)

        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)    
    else:             
        if type == 'text':
            text = body
            print('query: ', text)

            querySize = len(text)
            textCount = len(text.split())
            print(f"query size: {querySize}, words: {textCount}")

            if text == 'clearMemory':
                memory_chain.clear()
                map_chain[userId] = memory_chain
                    
                print('initiate the chat memory!')
                msg  = "The chat memory was intialized in this session."
            else:            
                if convType == 'normal':      # normal
                    msg = general_conversation(connectionId, requestId, chat, text)                  
                elif convType == 'langgraph-agent':
                    msg = run_langgraph_agent(connectionId, requestId, userId, text)      
                    
                    app = get_agent(userId)
                    config = {"configurable": {"thread_id": "thread-book"}}
                    current_state = app.get_state(config).values
                    print('current_state: ', current_state)
                    
                elif convType == 'bookstore-bot':
                    msg = run_bookstore_bot(connectionId, requestId, userId, app_bookstore, text)
                #elif convType == 'langgraph-agent-chat':
                #    msg = run_langgraph_agent_chat_using_revised_question(connectionId, requestId, chat, text)
                elif convType == 'plan-and-execute':
                    msg = run_plan_and_execute(connectionId, requestId, app_plan, text)
                else:
                    msg = general_conversation(connectionId, requestId, chat, text)  
                    
                memory_chain.chat_memory.add_user_message(text)
                memory_chain.chat_memory.add_ai_message(msg)
                                        
        elif type == 'document':
            isTyping(connectionId, requestId)
            
            object = body
            file_type = object[object.rfind('.')+1:len(object)]            
            print('file_type: ', file_type)
            
            if file_type == 'csv':
                docs = load_csv_document(path, doc_prefix, object)
                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
                        
            elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
                texts = load_document(file_type, object)

                docs = []
                for i in range(len(texts)):
                    docs.append(
                        Document(
                            page_content=texts[i],
                            metadata={
                                'name': object,
                                # 'page':i+1,
                                'uri': path+doc_prefix+parse.quote(object)
                            }
                        )
                    )
                print('docs[0]: ', docs[0])    
                print('docs size: ', len(docs))

                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
                
            elif file_type == 'py' or file_type == 'js':
                s3r = boto3.resource("s3")
                doc = s3r.Object(s3_bucket, s3_prefix+'/'+object)
                
                contents = doc.get()['Body'].read().decode('utf-8')
                
                #contents = load_code(file_type, object)                
                                
                msg = summary_of_code(chat, contents, file_type)                  
                
            elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
                print('multimodal: ', object)
                
                s3_client = boto3.client('s3') 
                    
                image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+object)
                # print('image_obj: ', image_obj)
                
                image_content = image_obj['Body'].read()
                img = Image.open(BytesIO(image_content))
                
                width, height = img.size 
                print(f"width: {width}, height: {height}, size: {width*height}")
                
                isResized = False
                while(width*height > 5242880):                    
                    width = int(width/2)
                    height = int(height/2)
                    isResized = True
                    print(f"width: {width}, height: {height}, size: {width*height}")
                
                if isResized:
                    img = img.resize((width, height))
                
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                commend  = jsonBody['commend']
                print('commend: ', commend)
                
                # verify the image
                msg = use_multimodal(img_base64, commend)       
                
                # extract text from the image
                text = extract_text(chat, img_base64)
                extracted_text = text[text.find('<result>')+8:len(text)-9] # remove <result> tag
                print('extracted_text: ', extracted_text)
                if len(extracted_text)>10:
                    msg = msg + f"\n\n[추출된 Text]\n{extracted_text}\n"
                
                memory_chain.chat_memory.add_user_message(f"{object}에서 텍스트를 추출하세요.")
                memory_chain.chat_memory.add_ai_message(extracted_text)
            
            else:
                msg = "uploaded file: "+object
            
                
        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)
        
        print('msg: ', msg)

        item = {
            'user_id': {'S':userId},
            'request_id': {'S':requestId},
            'request_time': {'S':requestTime},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg}
        }

        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            raise Exception ("Not able to write into dynamodb")               
        #print('resp, ', resp)

    return msg

def lambda_handler(event, context):
    # print('event: ', event)    
    msg = ""
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            #print("data[0:8]: ", body[0:8])
            if body[0:8] == "__ping__":
                # print("keep alive!")                
                sendMessage(connectionId, "__pong__")
            else:
                print('connectionId: ', connectionId)
                print('routeKey: ', routeKey)
        
                jsonBody = json.loads(body)
                print('request body: ', json.dumps(jsonBody))

                requestId  = jsonBody['request_id']
                try:
                    msg = getResponse(connectionId, jsonBody)
                    # print('msg: ', msg)
                    
                    sendResultMessage(connectionId, requestId, msg)  
                                        
                except Exception:
                    err_msg = traceback.format_exc()
                    print('err_msg: ', err_msg)

                    sendErrorMessage(connectionId, requestId, err_msg)    
                    raise Exception ("Not able to send a message")

    return {
        'statusCode': 200
    }