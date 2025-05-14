from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from docprocessing import vector_db
model=OllamaLLM(model="llama3.2")
QUERY_PROMPT = PromptTemplate(
    template="You are a helpful assistant. Answer the question based on the information provided.Question: {question}",
    input_variables=["question"],
)
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    model,
    prompt=QUERY_PROMPT,
)

template='''
 You are a helpful assistant and an expert in answering questions related to Postpartum Depression (PPD).
 Your task is to provide accurate and informative answers to the user's questions.
 Here is the question to answer: {question}'''

prompt=ChatPromptTemplate.from_template(template)
chain =({"context":retriever,"question": RunnablePassthrough()}
        |prompt
        | model
        | StrOutputParser()
)

while (True):
    question = input("Enter your question about Postpartum Depression (PPD): (q to quit) ")
    if question == 'q':
        break
    result=chain.invoke({"question":"what is postpartum depression?"})
    print(result)
