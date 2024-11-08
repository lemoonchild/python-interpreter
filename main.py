import streamlit as st
from langchain_experimental.tools import PythonREPLTool
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
import datetime
import os

load_dotenv()


def save_history(question, answer):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}: {question}-->{answer}\n")


def load_history():
    if os.path.exists("history.txt"):
        with open("history.txt", "r") as f:
            return f.readlines()

    return []


def main():
    st.set_page_config(page_title="Agente de Python Interactivo", page_icon="🤖", layout="wide")

    st.title("🤖 Agente de Python Interactivo")

    st.markdown(
        """
        <style>
        .stApp{background-color:black;}
        .title{color=#ff4b4b;}
        .button{background-color: #ff4b4b; color: white; border-radius: 5px;}
        .input{border: 1px solid #ff4b4b; border-radius: 5px;}
        </style> 
        """,
        unsafe_allow_html=True
    )

    instructions = """
    - Siempre usa la herramienta, incluso si sabes la respuesta.
    - Debes usar código de Python para responder.
    - Eres un agente que puede escribir código.
    - Solo responde la pregunta escribiendo código, incluso si sabes la respuesta.
    - Si no sabes la resppuesta escribe: "No sé la respuesta".
    """

    st.markdown(instructions)

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    st.write("Prompt loading...")

    tools = [PythonREPLTool()]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    st.markdown("### Ejemplos: ")

    ejemplos = [
        "Calcula la suma de 2 y 3",
        "Genera una lista del 1 al 10",
        "Crea una función que calcule el factorial de un número",
        "Crea un juego básico de snake con la librería pygame",
    ]

    example = st.selectbox("Selecciona un ejemplo: ", ejemplos)

    if st.button("Ejecutar ejemplo"):
        user_input = example

        try:
            answer = agent_executor.invoke(input={"input": user_input, "agent_scratchpad": ""})
            st.markdown("### Respuesta del agente: ")
            st.code(answer["output"], language="python")
            save_history(user_input, answer["output"])

        except ValueError as e:
            st.error(f"Error en el agente: {str(e)}")


if __name__ == "__main__":
    main()