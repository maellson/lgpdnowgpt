import streamlit as st
from utils import chatBot, text
from streamlit_chat import message


def main():

    st.set_page_config(page_title='LGPDNOW GPT', page_icon=':books:')

    st.header('Converse com um especialista')
    user_question = st.text_input("Em que posso te ajudar hoje?")

    if ('conversation' not in st.session_state):
        st.session_state.conversation = None

    if (user_question):

        response = st.session_state.conversation(user_question)['chat_history']

        for i, text_message in enumerate(response):

            if (i % 2 == 0):
                message(text_message.content,
                        is_user=True, key=str(i) + '_user')

            else:
                message(text_message.content,
                        is_user=False, key=str(i) + '_bot')

    with st.sidebar:

        st.subheader('Seus arquivos')
        pdf_docs = st.file_uploader(
            "Carregue os seus arquivos, em formato PDF, aqui", accept_multiple_files=True)

        if st.button('Processar'):
            all_files_text = text.process_file(pdf_docs)

            chunks = text.create_text_chunks(all_files_text)

            vectorstore = chatBot.create_vectorstore(chunks)
            print(vectorstore)

            st.session_state.conversation = chatBot.create_conversation_chain(
                vectorstore)


if __name__ == '__main__':

    main()
