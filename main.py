import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
from streamlit_chat import message
import PyPDF2
import os
from dotenv import load_dotenv

load_dotenv()


openai_api_key=os.getenv("OPENAI_API_KEY_2")
#This function will go through pdf and extract and return list of page texts.
def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        #print("Page Number:", len(pdfReader.pages))
        for i in range(len(pdfReader.pages)):
          pageObj = pdfReader.pages[i]
          text = pageObj.extract_text()
          pageObj.clear()
          text_list.append(text)
          sources_list.append(file.name + "_page_"+str(i))
    return [text_list,sources_list]
  
st.set_page_config(layout="centered", page_title="Multidoc_QnA")
# st.header("Multidoc_QnA")
# st.write("---")
  
#file uploader
uploaded_files = st.sidebar.file_uploader("Upload documents",accept_multiple_files=True, type=["txt","pdf"])
st.write("---")


def response(prompt, upload_files=uploaded_files):
    if uploaded_files is None:
        return st.info(f"""Upload files to analyse""")
    elif uploaded_files:
        st.write(str(len(uploaded_files)) + " document(s) loaded..")
    
        textify_output = read_and_textify(uploaded_files)
        
        documents = textify_output[0]
        sources = textify_output[1]
        
        #extract embeddings
        embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
        #vstore with metadata. Here we will store page numbers.
        vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
        #deciding model
        
        model_name = "gpt-3.5-turbo"
        # model_name = "gpt-4"

        retriever = vStore.as_retriever()
        retriever.search_kwargs = {'k':2}

        #initiate model
        llm = OpenAI(model_name=model_name, openai_api_key = openai_api_key, streaming=True)
        model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        result = model({"question":prompt}, return_only_outputs=True)

        return result['answer']
        # user_q = st.text_area("Enter your questions here")

if 'send_button' not in st.session_state:
    st.session_state['send_button'] = False
    # st.header("Ask Questions")

send_button=st.sidebar.button("Upload")

if send_button :
    st.session_state.send_button=True
    


if st.session_state.send_button==True:

    if "messages" not in st.session_state:

        st.session_state.messages=[]

    if 'generated' not in st.session_state:

        st.session_state['generated'] = []

        # chat_response = get_chat_response(st.session_state.messages)

        st.session_state.messages.append({"role":"assistant","content":"Hello"})

        st.session_state['generated'].append("hello")

    if 'past' not in st.session_state:

        st.session_state['past'] = []

        st.session_state['past'].append(" ")






    response_container = st.container()

    container = st.container()

 
 

    with container:

        placeholder = st.empty()

        with placeholder.form(key='my_form', clear_on_submit=True):

                user_input = st.text_input("You:", key='input')

                submit_button = st.form_submit_button(label='Send')

            

                # record_button=st.form_submit_button(label="record")


                if submit_button and user_input:
                    with st.spinner("Loading"):
                        st.session_state.messages.append({"role":"user","content":user_input})

                        print("messages dict-----",st.session_state.messages)

                        
                        
                        message_bot = response(user_input)
                        # st.subheader('Your response:')
                        # st.write(result['answer'])
                        # st.subheader('Source pages:')
                        # st.write(result['sources'])
                        # message_bot = get_chat_response(st.session_state.messages)

                        # message_bot = azure_gpt_call_davinci()

                        st.session_state.messages.append({"role":"assistant","content":message_bot},)

                        # st.sidebar.write(st.session_state.messages)

                        st.session_state['past'].append(user_input)

                        st.session_state['generated'].append(message_bot)
                    


    with response_container:

        if st.session_state['generated']:

            for i in range(len(st.session_state['generated'])):

                if st.session_state["past"][i]!=" ":

                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')

                    message(st.session_state["generated"][i], key=str(i))

 

                # else:

                #     message(st.session_state["generated"][i], key=str(i))
    

    # if st.button("Get Response"):
    #     try:
    #         with st.spinner("Model is working on it..."):
    #             result = model({"question":user_q}, return_only_outputs=True)
    #             st.subheader('Your response:')
    #             st.write(result['answer'])
    #             st.subheader('Source pages:')
    #             st.write(result['sources'])
    #     except Exception as e:
    #         st.error(f"An error occurred: {e}")
    #         st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
        
        
    
  
  

  
  
  
  
  
  
