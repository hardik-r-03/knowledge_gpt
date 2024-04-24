from typing import List
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document

from langchain.chat_models import ChatOpenAI
from knowledge_gpt.core.debug import FakeChatModel
from langchain.chat_models.base import BaseChatModel

from vertexai.generative_models import GenerativeModel, ChatSession
import vertexai
from vertexai import generative_models


def pop_docs_upto_limit(
    query: str, chain: StuffDocumentsChain, docs: List[Document], max_len: int
) -> List[Document]:
    """Pops documents from a list until the final prompt length is less
    than the max length."""

    token_count: int = chain.prompt_length(docs, question=query)  # type: ignore

    while token_count > max_len and len(docs) > 0:
        docs.pop()
        token_count = chain.prompt_length(docs, question=query)  # type: ignore

    return docs


def get_llm(model: str,project_id: str, **kwargs) -> BaseChatModel:
    # if model == "debug":
    #     return FakeChatModel()

    if model:
        project_id = "PROJECT_ID"
        location = "us-central1"
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel(model_name=model)
        chat = model.start_chat()
        return chat
    else:
        raise NotImplementedError(f"Model {model} not supported!")

def generate_answer(chat: ChatSession, context:str, query=query, chat_llmllm=chat_llm):
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)