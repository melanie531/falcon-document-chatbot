import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from typing import Dict
import json
from io import StringIO
from random import randint
from transformers import AutoTokenizer
from PIL import Image
import boto3
import numpy as np
import json

client = boto3.client('runtime.sagemaker')
def query_endpoint_with_json_payload(encoded_json, endpoint_name):
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=encoded_json)
    return response

def parse_response(query_response):
    response_dict = json.loads(query_response['Body'].read())
    return response_dict['generated_images'], response_dict['prompt']

st.set_page_config(page_title="Document Analysis", page_icon=":robot:")


endpoint_names = {
    "Falcon-40b-instruct":"falcon-40b-instruct-12xl",
    "Flan-T5-XXL":"flan-t5-xxl-12xl",
    "Stable Diffusion 2.1":"jumpstart-dft-stable-diffusion-v2-1-base"
}



class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    len_prompt = 0

    def transform_input(self, prompt: str, model_kwargs: Dict={}) -> bytes:
        self.len_prompt = len(prompt)
        input_str = json.dumps({"inputs": prompt, "parameters":{"max_new_tokens": st.session_state.max_token, "temperature":st.session_state.temperature, "seed":st.session_state.seed, "stop": ["Human:"]}})
        print(input_str)
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = output.read()
        res = json.loads(response_json)
        print(res)
        ans = res[0]['generated_text'][self.len_prompt:]
        ans = ans[:ans.rfind("Human")].strip()
        
        return ans
    
class ContentHandlerT5(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    len_prompt = 0

    def transform_input(self, prompt: str, model_kwargs: Dict={}) -> bytes:
        self.len_prompt = len(prompt)
        input_str = json.dumps({"inputs": prompt, "parameters":{"max_length": st.session_state.max_token, "temperature":st.session_state.temperature, "seed":st.session_state.seed, "stop": ["Human:"]}})
        print(input_str)
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = output.read()
        res = json.loads(response_json)
        print(res)
        ans = res[0]['generated_text']
        
        return ans    


    
content_handler = ContentHandler()
content_handler_t5 = ContentHandlerT5()


@st.cache_resource
def load_chain(endpoint_name: str="falcon-40b-instruct-12xl"):
    if endpoint_name == "falcon-40b-instruct-12xl":
        llm = SagemakerEndpoint(
                endpoint_name=endpoint_name,
                region_name="us-east-1",
                content_handler=content_handler,
        )
    else:
        llm = SagemakerEndpoint(
                endpoint_name=endpoint_name,
                region_name="us-east-1",
                content_handler=content_handler_t5,
        )
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)
    return chain

chatchain = load_chain()


# initialise session variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
    chatchain.memory.clear()
if 'widget_key' not in st.session_state:
    st.session_state['widget_key'] = str(randint(1000, 100000000))
    
def clear_button_fn(option):
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['widget_key'] = str(randint(1000, 100000000))
    chatchain = load_chain(endpoint_name=endpoint_names[option])
    chatchain.memory.clear()
    
def on_status_change():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['widget_key'] = str(randint(1000, 100000000))
    chatchain.memory.clear()

with st.sidebar:
    # Sidebar - the clear button is will flush the memory of the conversation
    st.sidebar.title("Conversation setup")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")


    # upload file button
    uploaded_file = st.sidebar.file_uploader("Upload a txt file", type=["txt"], key=st.session_state['widget_key'])
    
    option = st.selectbox(
        "Which model you would like to use?",
        ("Falcon-40b-instruct", "Flan-T5-XXL", "Stable Diffusion 2.1"),
        on_change=on_status_change
    )
    
    if option != "Stable Diffusion 2.1":
        chatchain = load_chain(endpoint_name=endpoint_names[option])
    
    if clear_button:
        clear_button_fn(option)
    
        
    

left_column, _, right_column = st.columns([50, 2, 20])

with left_column:
    st.header("Building a chatbot with Amazon SageMaker Jumpstart Foundation Models")
    # this is the container that displays the past conversation
    response_container = st.container()
    # this is the container with the input text box
    container = st.container()
    
    with container:
        # define the input text box
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Input text:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')
        if option != "Stable Diffusion 2.1":   
            # when the submit button is pressed we send the user query to the chatchain object and save the chat history
            if submit_button and user_input:
                output = chatchain(user_input)["response"]
                print(output)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
            # when a file is uploaded we also send the content to the chatchain object and ask for confirmation
            elif uploaded_file is not None:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                content = "=== BEGIN FILE ===\n"
                content += stringio.read().strip()
                content += "\n=== END FILE ===\nPlease confirm that you have read that file by saying 'Yes, I have read the file'"
                output = chatchain(content)["response"]
                st.session_state['past'].append("I have uploaded a file. Please confirm that you have read that file.")
                st.session_state['generated'].append(output)

            history = chatchain.memory.load_memory_variables({})["history"]
        else:
            if submit_button and user_input:
                payload = { "prompt": user_input, "width":400, "height":400,
               "num_images_per_prompt":1, "num_inference_steps":50, "guidance_scale":7.5, "seed":st.session_state.seed}
                print(payload)
                query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'),endpoint_names[option])
                generated_images, prompt = parse_response(query_response)
                image = np.array(generated_images)
                st.image(image, output_format="auto", caption=prompt)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(f"Generated an image of {user_input}")
        st.write(f"Currently using model: {option}")


    # this loop is responsible for displaying the chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                
with right_column:
    st.header('Available parameters:')
    max_new_tokens= st.slider(
        min_value=8,
        max_value=1024,
        step=1,
        value=150,
        label="Number of tokens to generate",
        key="max_token"
    )
    temperature = st.slider(
        min_value=0.1,
        max_value=2.5,
        step=0.1,
        value=0.3,
        label="Temperature",
        key="temperature"
    )
    seed = st.slider(
        min_value=0,
        max_value=1000,
        step=1,
        label="Random seed to use for the generation",
        key="seed"
    )
