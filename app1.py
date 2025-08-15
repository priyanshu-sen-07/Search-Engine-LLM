import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler  # ✅ updated import

# -------------------------
# App Title
# -------------------------
st.title("🔎 LangChain - Chat with Search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions 
of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at:
[github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent)
"""

# -------------------------
# Sidebar Settings
# -------------------------
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# -------------------------
# Tools Setup
# -------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# -------------------------
# Session State Messages
# -------------------------
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# -------------------------
# Chat Input
# -------------------------
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="Llama3-8b-8192",
            streaming=True
        )
        
        tools = [search, arxiv, wiki]

        # ✅ Correct parameter name: handle_parsing_errors
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                # ✅ Only pass user query, not full history
                response = search_agent.run(prompt, callbacks=[st_cb])
            except Exception as e:
                response = f"⚠️ Error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
