import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
import os
from dotenv import load_dotenv
load_dotenv()


from phi.embedder.sentence_transformer import SentenceTransformerEmbedder

embeddings = SentenceTransformerEmbedder()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes",db_url=db_url, embedder=embeddings),

    )

knowledge_base.load()

storage = PgAssistantStorage(table_name='pdf_assistant', db_url=db_url)

def pdf_assistant(new: bool = False, user: str = 'user'):

    run_id: Optional[str] = None
    if not new:
        existing_run_ids: List[str] = storage.get_all_ids(user)

        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id = run_id,
        user_id = user,
        knowledge_base = knowledge_base,
        storage = storage,
        # show tool calls in the response
        show_tool_calls = True,
        # enable the assistant to search the knowledge base
        search_knowledge=True,
        # enable the assistant to read thechat history
        read_chat_history=True
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f'Started Run: {run_id}\n')
    else:
        print(f'Continuing Run: {run_id}\n')

    assistant.cli_app(markdown = True)

if __name__ == '__main__':
    typer.run(pdf_assistant)





































# import typer
# from typing import Optional, List
# from phi.assistant import Assistant
# from phi.storage.assistant.postgres import PgAssistantStorage
# from phi.knowledge.pdf import PDFUrlKnowledgeBase
# from phi.vectordb.pgvector import PgVector  # ✅ Use PgVector instead of PgVector2
# from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # ✅ Use HuggingFace sentence-transformer embedder
# embedder = SentenceTransformerEmbedder()

# # ✅ Database URL (PostgreSQL)
# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# # ✅ Create knowledge base with local HuggingFace embeddings
# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=PgVector(  # ✅ Use PgVector
#         db_url=db_url,
#         table_name="sentence_transformer_embeddings",  # You can name it anything
#         embedder=embedder,
#     ),
#     num_documents=2,  # Optional: number of chunks to keep
# )

# # ✅ Load knowledge base
# knowledge_base.load()

# # ✅ Set up storage
# storage = PgAssistantStorage(table_name='pdf_assistant', db_url=db_url)

# def pdf_assistant(new: bool = False, user: str = 'user'):
#     run_id: Optional[str] = None
#     if not new:
#         existing_run_ids: List[str] = storage.get_all_ids(user)
#         if existing_run_ids:
#             run_id = existing_run_ids[0]

#     assistant = Assistant(
#         run_id=run_id,
#         user_id=user,
#         knowledge_base=knowledge_base,
#         storage=storage,
#         show_tool_calls=True,
#         search_knowledge=True,
#         read_chat_history=True,
#     )

#     if run_id is None:
#         print(f'Started Run: {assistant.run_id}\n')
#     else:
#         print(f'Continuing Run: {run_id}\n')

#     assistant.cli_app(markdown=True)

# if __name__ == '__main__':
#     typer.run(pdf_assistant)