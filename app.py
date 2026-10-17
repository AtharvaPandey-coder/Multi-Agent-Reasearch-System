import streamlit as st
from reasearch_assistant import build_retriever,summarising_final_answer,build_agent
from dotenv import load_dotenv
import requests
load_dotenv()

st.set_page_config('Multi Tool reasearch Assistant')
st.title('Multi Tool reasearch Assistant')
st.caption('upload the document(optional) .the agent tries it first,then falls back to web search or wikipedia search if needed')

#Step1->Optional
uploaded_file=st.file_uploader("Uploads a PDF(optional)",type=['pdf'])

#Step2 -> Build retriever + agent once per session
if uploaded_file is not None and "agent" not in st.session_state:
    with st.spinner('Processing document ....'):
        temp_path=f"temp_{uploaded_file.name}"
        with open(temp_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        retriever=build_retriever(temp_path)
        st.session_state.agent=build_agent(retriever)
    st.success('document Indexed. Ask away.')

if uploaded_file is None and "agent"  not  in st.session_state:
    st.session_state.agent=build_agent(retriever=None)

#Step-3:Question
question=st.text_area('Enter the Question')

if st.button("Run") and question:
    with st.spinner('Agent is thinking'):
        result=st.session_state.agent.invoke(
        {'messages':[{'role':"user",'content':question}]}
        )
        agent_output=result['messages'][-1].content
        final_answer =summarising_final_answer(question,agent_output)

    st.subheader("Answer")
    st.write(final_answer)

    with st.expander("See agent's reasoning"):
        for m in result['messages']:
            if getattr(m,"type","")=="tool":
                st.text(m.content[:500])



