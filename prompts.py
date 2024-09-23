RAG_MSG="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise and output in a short answer form.

Question: {question} 

Context: {context} 

Answer:
"""

NO_CONTEXT_MSG="""You are an assistant for question-answering tasks. If you don't know the answer, just say that you don't know. Keep the answer concise and output in a short answer form.

Question: {question} 

Answer:
"""



QUESTIONDECOMPOSE_MSG="""You are an assistant for question decomposing. Decompose the given a multi-hop question into a sub question that could be solved in single hop. Try to minimize the number of sub questions. Also return your sub questions in a strict format of a single python list containing strings with sub question surround with \".

Question: {question}

Sub questions:
"""


EVIDENCEEXTRACT_MSG="""You are an assistant for extracting evidence. Choose the best context that helps you to solve the given question. You should only pick a single context. Also return your choice in a number that indicates the passage, which are numerical values.

Question: {question}

Context: {context}

Best context:
"""

EVIDENCEEXTRACT_MULTI_MSG="""You are an assistant for extracting evidence. Choose the best context that helps you to solve the given question. Also return your choice in a format of python list containing number that indicates the passage, which are numerical values.

Question: {question}

Context: {context}

Best context:
"""

