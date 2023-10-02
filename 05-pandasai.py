import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import Falcon

load_dotenv()


def main():
    st.set_page_config(page_title="Ask your CSV")
    st.title("Chat with CSV using PandasAI 🐼")

    uploaded_file = st.file_uploader("Upload your Data", type="csv")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        llm = Falcon(api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'])

        df = pd.read_csv(tmp_file_path)
        sdf = SmartDataframe(df, config={"llm": llm})

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(sdf.chat(user_question))


if __name__ == "__main__":
    main()
