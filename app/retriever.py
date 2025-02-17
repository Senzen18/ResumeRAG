from pydantic import BaseModel,Field
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.schema.agent import AgentFinish
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
class JobDescription(BaseModel):
    """Description of job for retreiving resumes"""
    job_description: str = Field(... , description='Description of job to retreive similar resumes from vector database')
class ApplicantID(BaseModel):
    """'List of applicant ids to retreive resumes for'"""
    ids_list: list = Field(... , description='List of applicant ids to retreive resumes for')
    
class RAGRetriever:
    def __init__(self,vectordb,df):
        self.df = df
        self.vectordb = vectordb
    def  __reciprocal_rank_fusion__(self,document_ids_list,k=50):
        fused_scores = {}
        for doc_list in document_ids_list:
            for rank,doc_id in enumerate(doc_list.keys()):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1/(rank+k)
        reranked_results = {doc_id: scores for doc_id,scores in sorted(fused_scores.items(), key = lambda x: x[1], reverse=True)}
        return reranked_results
                
        
    def __retreive_ids__(self,question,k=50):
        retreived_docs = self.vectordb.similarity_search_with_score(question,k=k)
        retreived_docs = {docs.metadata['ids']: scores for docs,scores in retreived_docs}
        return retreived_docs
    def retreive_rerank(self,subquestions_list):
        document_ids_list = []
        for subquestion in subquestions_list:
            document_ids_list.append(self.__retreive_ids__(subquestion))
        reranked_list = self.__reciprocal_rank_fusion__(document_ids_list)
        return reranked_list
    def retreive_docs_from_ids(self,id_to_score,threshold=5):
        resumes_dict = dict(zip(self.df['ids'].astype(str),self.df['resume']))
        ids_list = [str(doc_ids) for doc_ids,scores in sorted(id_to_score.items(),key = lambda x:x[1], reverse = True)][:threshold]
        #print(ids_list)
        docs_list = [resumes_dict[ids] for ids in ids_list]
        for i in range(len(docs_list)):
            docs_list[i] = "Applicant ID:" + ids_list[i] + "\n" + docs_list[i]
        return docs_list
            
class QueryRetriever(RAGRetriever):
    def __init__(self,vectordb,df):
        super().__init__(vectordb,df)
        self.prompt = ChatPromptTemplate.from_messages(
            [('system','You are a member of talent acquisition/HR'),
            ("user","{input}")]
        )
    def retreive_resume(self,question,llm = ChatOpenAI(temperature=0),rag_method = "RAG"):
        @tool(args_schema=JobDescription)
        def retreive_resume_jd(job_description:str):
            """Retrieve Resumes similar to the given job description"""
            query_list = [job_description]
            if rag_method == 'rag_fusion':
                query_list += llm.generate_subqueries(job_description)
            #print(query_list)
            doc_ids = self.retreive_rerank(query_list)
            resumes = self.retreive_docs_from_ids(doc_ids)
            return resumes
        @tool(args_schema=ApplicantID)
        def retreive_resume_id(ids_list:list):
            """Retrieve Resumes similar to the given list of ID's"""
            try:
                ids_list = [str(ids) for ids in ids_list]
                resumes_dict = dict(zip(self.df['ids'].astype(str),self.df['resume']))
                docs_list = [resumes_dict[ids] for ids in ids_list]
                for i in range(len(docs_list)):
                    docs_list[i] = "Applicant ID:" + ids_list[i] + "\n" + docs_list[i]
        
            except:
                return []
            return docs_list
        def Router(response):
            if isinstance(response,AgentFinish):
               return response.return_values['output']
            else:
                toolbox = {'retreive_resume_jd':retreive_resume_jd,
                           'retreive_resume_id':retreive_resume_id}
                return toolbox[response.tool].run(response.tool_input)
        
        llm_func_call = llm.llm.bind(functions = [format_tool_to_openai_function(tool) for tool in [retreive_resume_id,retreive_resume_jd]])
        chain = self.prompt | llm_func_call | OpenAIFunctionsAgentOutputParser() | Router
        result = chain.invoke({'input':question})
        return result
        
            
            
    