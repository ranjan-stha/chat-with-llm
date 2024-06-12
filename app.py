import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from llm_app import llm

MAX_SIZE = 20

st.title("Interact with LLM Model")

def get_response(query: str, chat_history: list):
    template = """
        You are a helpful assistant. Answer the query based on the chat history
        Chat History: {chat_history}
        User Query: {query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "query": query
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

user_query = st.chat_input("Enter your query !!")

if user_query:
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(ai_response))

    if len(st.session_state.chat_history) > MAX_SIZE:
        del st.session_state.chat_history[0]
