import streamlit as st

from knowledge_gpt.components.sidebar import sidebar

from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from knowledge_gpt.core.caching import bootstrap_caching

from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm
from knowledge_gpt.core.utils import generate_answer

from pathlib import Path
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gemini-1.0-pro-vision"]

# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")

st.set_page_config(page_title="Maxima Patient Assist", page_icon="👩🏼‍🔬", layout="wide")
st.header("👩🏼‍🔬Maxima Patient Assist")

# Enable caching for expensive functions
bootstrap_caching()

# sidebar()

# openai_api_key = st.session_state.get("OPENAI_API_KEY")


# if not openai_api_key:
#     st.warning(
#         "Enter your OpenAI API key in the sidebar. You can get a key at"
#         " https://platform.openai.com/account/api-keys."
#     )


uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt", "jpg", "png"],
    help="Scanned documents are not supported yet!\n PLease enter one of "pdf", "docx", "txt", "jpg", "png"",
)

model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

with st.expander("Advanced Options"):
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")


if not uploaded_file:
    st.stop()


# Read uploaded image
if Path(uploaded_file.name).suffix == "jpg" or Path(uploaded_file.name).suffix == "png":
    try:
        image = generative_models.Part.from_uri(Path(uploaded_file.name), mime_type="image/jpeg")
    except Exception as e:
        print("Error in reading uploaded image")
        print(e)

# Read uploaded document
if Path(uploaded_file.name).suffix in ["doc", "txt", "docx", "csv"]:
    try:
        file = read_file(uploaded_file)
    except Exception as e:
        display_file_read_error(e, file_name=uploaded_file.name)

# not needed
# chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

with st.spinner("Reading document... This may take a while⏳"):
    gemini_pro_vision_model = GenerativeModel(model)
    file_data = gemini_pro_vision_model.generate_content(
        ["Take a close look and Examine the image carefully. Generate a descriptive summary of the type of details like medicine names, schedules, dates, clinic name, phramacy name, patient details, etc", image])


if not is_file_valid(file):
    st.stop()


# if not is_open_ai_key_valid(openai_api_key, model):
#     st.stop()



    # folder_index = embed_files(
    #     files=[chunked_file],
    #     embedding=EMBEDDING if model != "debug" else "debug",
    #     vector_store=VECTOR_STORE if model != "debug" else "debug",
    #     openai_api_key=openai_api_key,
    # )

with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")


if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)


if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col = st.columns(1)

    chat_llm = get_llm(model=model, project_id="q-gcp-00109-pm-hackathon-24-04", temperature=0)
    result = generate_answer(context=file_data, query=query, chat_llm=chat_llm)

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result.answer)

    # with sources_col:
    #     st.markdown("#### Sources")
    #     for source in result.sources:
    #         st.markdown(source.page_content)
    #         st.markdown(source.metadata["source"])
    #         st.markdown("---")
