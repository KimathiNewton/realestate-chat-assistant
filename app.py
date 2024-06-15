__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

import csv
import json
import os
import urllib
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, \
    PromptTemplate
from langchain.schema import Document
from langchain.vectorstores import Chroma

from prompts import (
    condense_question_chain_system_prompt,
    condense_question_chain_human_prompt,
    ai_prompt
)

load_dotenv()

# llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))
ikigai_license_key = os.environ.get('IKIGAI_LICENSE_KEY')

DATA_DIR = "./data"
IKIGAI_INFO_COLLECTION_NAME = "ikigai" # noqa
IKIGAI_FAQS_COLLECTION_NAME = "ikigai_faqs"
PERSISTENCE_DIR_PATH = "./chroma_db"

DATA_DIR_PATH = "./data"

def get_ikigAi_info():
    ikigAI_url = "https://ikigaitech.io/wp-json/ikigai-api/v1/get-bot-info"
    ikigAI_params = {'license_key': ikigai_license_key, 'product': 'chat'}
    ikigAi_response = requests.post(url=ikigAI_url, params=ikigAI_params)

    # Check we have a valid response for
    if ikigAi_response.status_code != 200:
        st.error(f'Unable to retrieve AI information. Please contact support. License Key: {ikigai_license_key}',
                 icon="🚨")
        exit()

    ikigAi_info = ikigAi_response.json()

    if not ikigAi_info['valid']:
        st.error(f'Invalid key.. Please contact support. License Key: {ikigai_license_key}', icon="🚨")
        exit()

    return ikigAi_info


def convert_csvs_to_jsons(directory_path=DATA_DIR):
    def remove_specific_keys(row):
        keys_to_remove = ["development_contact_number",
                          "development_location_postcode_override", "no_of_receptions", "development_opening_times"]
        return {k: v for k, v in row.items() if k not in keys_to_remove}

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(directory_path, filename)
            json_file_path = os.path.join(directory_path, filename.replace('.csv', '.json'))

            # Convert CSV to JSON
            with open(csv_file_path, mode='r', encoding='utf-8') as csv_file, \
                    open(json_file_path, mode='w', encoding='utf-8') as json_file:
                csv_reader = csv.DictReader(csv_file)
                data = [remove_specific_keys(row) for row in csv_reader]
                json.dump(data, json_file, indent=4)


def get_json_loader(filepath):
    print("getting loader for :", filepath)
    return JSONLoader(file_path=filepath, jq_schema=".[]", text_content=False)


def get_ikigai_faqs():
    ikigai_faqs_url = f"https://ikigaitech.io/wp-content/uploads/ikigai_chat/faqs/{ikigai_license_key}-faqs.json"
    response = requests.get(ikigai_faqs_url)

    if response.status_code != 200:
        st.error(f"Unable to fetch faq's. License key: {ikigai_license_key}", icon="🚨")
        exit()

    return json.loads(response.text)


def create_embeddings_for_ikigai_faqs_questions(ikigai_faqs):
    ikigai_faq_documents = [
        Document(page_content=ikigai_faq.get("question"), metadata={"answer": ikigai_faq.get("answer")})
        for ikigai_faq in ikigai_faqs
    ]
    chroma = Chroma.from_documents(
        collection_name=IKIGAI_FAQS_COLLECTION_NAME,
        documents=ikigai_faq_documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSISTENCE_DIR_PATH,
    )
    chroma.persist()


def semantic_search_on_faqs(query):
    vector_store = Chroma(
        collection_name=IKIGAI_FAQS_COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=PERSISTENCE_DIR_PATH,
    )
    retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8}
        )

    return retriever.get_relevant_documents(query)


def validate_chroma_db_collection(collection_name):
    collection_exists = False
    client = chromadb.PersistentClient(PERSISTENCE_DIR_PATH)
    collection_names = [collection.name for collection in client.list_collections()]

    if collection_name in collection_names:
        collection_exists = True

    return collection_exists


def create_chunks_of_files_content():
    docs = []
    json_loaders = [
        get_json_loader(os.path.join("./data", filename))
        for filename in os.listdir("./data")
        if filename.endswith(".json")
    ]
    for json_loader in json_loaders:
        docs.extend(json_loader.load())

    return docs


# Converts a single record from the Actor's resulting dataset to the LlamaIndex format
def transform_dataset_item(item):
    return Document(
        text=item.get("text") + " - Find out more info at this Source URL: " + item.get("url"),
        extra_info={
            "url": item.get("url"),
        },
    )


# Safe creation of directories
def safe_open_w(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')


def remove_image_files(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpeg', '.jpg', '.png', '.html', '.xml')):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)


def create_collection():
    convert_csvs_to_jsons()
    docs = create_chunks_of_files_content()
    print("created Docs:", len(docs))

    chroma = Chroma.from_documents(
        collection_name=IKIGAI_INFO_COLLECTION_NAME,
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=PERSISTENCE_DIR_PATH,
    )
    chroma.persist()


def delete_collection(collection_name):
    client = chromadb.PersistentClient(PERSISTENCE_DIR_PATH)
    client.delete_collection(collection_name)

# Load Data
@st.cache_resource(show_spinner=False)
def load_data(ikigAi_info):
    ikigAI_sitemap = (
        f'https://ikigaitech.io/wp-content/uploads/ikigai_chat/sitemaps/{ikigai_license_key}-sitemap.html')
    ikigAI_csv = (f'https://ikigaitech.io/wp-content/uploads/ikigai_chat/csv/{ikigai_license_key}-data.csv')
    ikigAI_crawl = (f'https://api.apify.com/v2/datasets/{ikigAi_info["info"]["apify_id"]}/items')
    ikigAI_additional_sources = json.loads(ikigAi_info["info"]["additional_sources"])

    with st.spinner(
            text="We're just getting the latest info from our website – hang tight! This should take 1-2 minutes."):

        # Get Sitemap, Crawl and Data
        urllib.request.urlretrieve(ikigAI_crawl, "./data/crawl.json")
        urllib.request.urlretrieve(ikigAI_sitemap, "./data/sitemap.html")
        urllib.request.urlretrieve(ikigAI_csv, "./data/data.csv")

        # Get any additonal sources
        if ikigAI_additional_sources:
            for x in ikigAI_additional_sources:
                additional_source = urlparse(x)
                urllib.request.urlretrieve(x, "./data/" + os.path.basename(additional_source.path))

    create_collection()


def rephrase_question():
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
    cq_question_system_prompt_template = SystemMessagePromptTemplate.from_template(
        condense_question_chain_system_prompt)
    cq_question_human_prompt_template = HumanMessagePromptTemplate.from_template(condense_question_chain_human_prompt)
    cq_question_chat_prompt = ChatPromptTemplate.from_messages(
        [cq_question_system_prompt_template, cq_question_human_prompt_template])
    condense_question_chain = LLMChain(llm=ChatOpenAI(temperature=0, model="gpt-4"), prompt=cq_question_chat_prompt,
                                       verbose=True)

    return condense_question_chain


if "question_rephraser_chain" not in st.session_state.keys():
    st.session_state.question_rephraser_chain = rephrase_question()


def get_relevant_sources(query):
    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(),
        collection_name=IKIGAI_INFO_COLLECTION_NAME,
        persist_directory=PERSISTENCE_DIR_PATH,
    )
    #retriever = vector_store.as_retriever(
        #search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1, "k": 10}
        # search_type="similarity", search_kwargs={"k": 10}
    #)
    qa = RetrievalQAWithSourcesChain.from_chain_type(
    OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=openai_api_key), chain_type="stuff", retriever=docstorage.as_retriever()
)
    sources = qa.get_relevant_documents(query)
    pages_content = [source.page_content for source in sources]
    return pages_content


def get_chat_answer(sources, query):
    conversation_prompt = PromptTemplate.from_template(ai_prompt)
    conversation_chain = LLMChain(
        prompt=conversation_prompt, llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106"), verbose=True
    )

    return conversation_chain.run(query=query, sources=sources)


def post_chat_history(ikigai_license_key, ikigai_chat_user_id, role, message):
    ikigAI_url = "https://ikigaitech.io/wp-json/ikigai-api/v1/post-chat-history"
    ikigAI_params = {
        'license_key': ikigai_license_key, 
        'product': 'chat', 
        'ikigai_chat_user_id': ikigai_chat_user_id,
        'role' : role,
        'message' : message
    }
    ikigAI_response = requests.post(url=ikigAI_url, params=ikigAI_params);
    return ikigAI_response

def post_form_submission(ikigai_license_key, ikigai_chat_user_id, form_key, form_data):
    ikigAI_url = "https://ikigaitech.io/wp-json/ikigai-api/v1/post-form-submission"
    ikigAI_params = {
        'license_key': ikigai_license_key, 
        'product': 'chat', 
        'ikigai_chat_user_id': ikigai_chat_user_id,
        'form_key' : form_key,
        'form_data' : json.dumps(form_data)
    }
    ikigAI_response = requests.post(url=ikigAI_url, params=ikigAI_params);
    return ikigAI_response

ikigAi_info = get_ikigAi_info()

#st.error(f'Chat Prompt: {ikigAi_info}',icon="🚨")
#exit()

# Check we have a successful response for
st.set_page_config(page_title=ikigAi_info['info']['chat_title'], page_icon="", layout="wide",
                   initial_sidebar_state="auto", menu_items=None)
st.title(ikigAi_info['info']['chat_title'])

# Add custom styling
ikigAI_custom_html = ikigAi_info['info']['custom_html']
st.markdown(f'{ikigAI_custom_html}', unsafe_allow_html=True)

# Get User ID or Prompt if exists
ikigai_chat_params = st.experimental_get_query_params();
ikigai_chat_user_id = ikigai_chat_params.get("ikigai_id")[0]

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": ikigAi_info['info']['chat_prompt']}
    ]

    # Build back up the context from old messages (if they exist)
    ikigAI_url = "https://ikigaitech.io/wp-json/ikigai-api/v1/get-previous-chats"
    ikigAI_params = {'license_key': ikigai_license_key, 'product': 'chat', 'ikigai_chat_user_id': ikigai_chat_user_id}
    ikigAI_response = requests.post(url=ikigAI_url, params=ikigAI_params)

    # Check we have a valid response for
    if ikigAI_response.status_code == 200 and ikigAi_info['valid']:

        ikigAI_previous_chats = ikigAI_response.json()

        if ikigAI_previous_chats['chats']:
            for chat_message in ikigAI_previous_chats['chats']:
                st.session_state.messages.append({"role": chat_message['type'], "content": chat_message['message']})

        # Pass new chat prompt if needed
        if ikigai_chat_params.get("prompt"):
            st.session_state.messages.append({"role": "user", "content": ikigai_chat_params.get("prompt")[0]})

# Create data directory
with safe_open_w('./data/data.txt') as f:
    f.write('Creating folder')

# Initialize the chat engine
load_data(ikigAi_info)

if int(ikigAi_info["info"].get("faqs")):
    # will implement it if there is duplication of embeddings
    # if validate_chroma_db_collection(IKIGAI_FAQS_COLLECTION_NAME):
    #     delete_collection(IKIGAI_FAQS_COLLECTION_NAME)

    # Store faq's in vector database
    ikigai_faqs = get_ikigai_faqs()
    create_embeddings_for_ikigai_faqs_questions(ikigai_faqs)

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":

    post_chat_history(ikigai_license_key, ikigai_chat_user_id, st.session_state.messages[-1]["role"], st.session_state.messages[-1]["content"])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get the response
            similar_faq = None
            condensed_question = st.session_state.question_rephraser_chain.run(
                question=prompt,
                chat_history=st.session_state.messages[:-1][-5:] if len(
                    st.session_state.messages) > 4 else st.session_state.messages[1:-1]
            )

            if validate_chroma_db_collection(IKIGAI_FAQS_COLLECTION_NAME):
                similar_faq = semantic_search_on_faqs(condensed_question)

            if similar_faq:
                answer = similar_faq[0].metadata.get("answer")
            else:
                sources = get_relevant_sources(condensed_question)
                answer = get_chat_answer(sources, condensed_question)

            # Display response
            st.write(answer)
            message = {"role": "assistant", "content": answer}

            # Add response to message history
            st.session_state.messages.append(message)

            # Post to DB
            post_chat_history(ikigai_license_key, ikigai_chat_user_id, "assistant", answer)

# Show Enquiry Form
formbtn = st.button("Form")

if "formbtn_state" not in st.session_state:
    st.session_state.formbtn_state = False

if formbtn or st.session_state.formbtn_state:
    st.session_state.formbtn_state = True
    form_key = "enquiry_form"
    st.subheader("Enquiry Form")
    with st.form(key = form_key):
        st.write('Let us know how we can help')
    
        name = st.text_input(label="Name")
        email = st.text_input(label="Email")
    
        submit_form = st.form_submit_button(label="Send Enquiry")
    
        # Checking if all the fields are non empty
        if submit_form:

            if name and email:
                post_form_submission(ikigai_license_key, ikigai_chat_user_id, form_key, [name, email])
                st.success("Thank you for your enquiry.")
            else:
                st.warning("Please fill all the fields")
