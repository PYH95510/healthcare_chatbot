<a href="https://club-project-one.vercel.app/" target="_blank">
</a>

<br/>
<br/>

# 0. Getting Started (시작하기)
Consisted with two major datasets. In total, there are 237,000 columns.


<br/>
<br/>

# 1. Project Overview (프로젝트 개요)
- Project Name(프로젝트 이름): Healthcare_chatbot
- Project Explaination(프로젝트 설명): Using various dataset to compose RAG for improving quality of response

<br/>
<br/>

# 2. Team Members (팀원 및 팀 소개)
| By myself |

<br/>
<br/>

# 3. Key Features (주요 기능)
- **Choosing Response Style(답변 스타일 선택)**:
  - Standard (Standard chatbot style answers)
  - Chain of Thought (Applying COT template to adjust quality response)

- **Retrieving Chat History**:
  - While answering, LLM model will retrieve history of the whole chat

- **RAG Ranking**:
  - By using RAG index, the LLM will produce quality, up-to-date answer.

- **LLM as a Judge**:
  - Using LLM as a judge for the response, quality answer will be produced.
 
- **Two Distinct Servers for Fast and Better Response**:
  - With two distinct servers, I was able to come up with fast, latent free response server as well as concrete judge server.
 
- **Various Prompting**:
  - Standard prompting for general usage.
  - COT(Chain of Thought) prompting for better chatbot response.
  - Prompting for LLM as a judge.

<br/>
<br/>

# 4. Technology Stack (기술 스택)
## 4.1 Language
|  |  |  |
|-----------------|-----------------|-----------------|
| Python    |  <img src="https://github.com/user-attachments/assets/621c83cd-eb45-45e9-ac1a-582ea6771dd8" alt="Python" width="100"> | 3.11    |


<br/>

## 4.2 Frotend
|  |  |  |
|-----------------|-----------------|-----------------|
| Streamlit    |  <img src="https://github.com/user-attachments/assets/683a1aef-4289-49a9-9306-1843ff3f6088" alt="Streamlit" width="100"> | 1.42.0    |


<br/>

## 4.3 Backend
|  |  |  |
|-----------------|-----------------|-----------------|
| FastAPI    |  <img src="https://github.com/user-attachments/assets/56acaf71-6fcd-4cb6-bf14-fb0668552c91" alt="FastAPI" width="100">    | 0.115.8    |

<br/>

## 4.4 ETC.
## Pytorch, Numpy, Pandas, Sentence Transformer, Hugging Face, Ollama (Llama3.2 1B-Instruct), Conda(venv), 
<br/>
<br/>

# 5. Project Structure (프로젝트 구조)
```plaintext
project/
├── .DS_Store
├── .gitignore               # Git 무시 파일 목록
│   LICENSE
│   RAG.py    # 정확한 종속성 버전이 기록된 파일로, 일관된 빌드를 보장
│   README.md                # Project Overview(프로젝트 개요 및 사용법)
│   buildingFaissIndex.py         # only for building Faiss index and testing it.
│   chatbot.py         # Streamlit instantiated, trimming answers for users.
│   chatbot_evaluate.py         # evaluating chatbot performance.
│   data prep.py         # showing how data was prepared.
│   judge_llm_server.py         # server for LLM judge
│   llm_server.py         # server for LLM 
└── prompt.py         # prompt for various usages.
```

<br/>
<br/>

# 6. Dataset Columns (데이터셋 컬럼 설명)
id : a string question identifier for each example
question : question text (a string)
opa : Option A
opb : Option B
opc : Option C
opd : Option D
cop : Correct option, i.e., 1,2,3,4
choice_type ({"single", "multi"}): Question choice type.
"single": Single-choice question, where each choice contains a single option.
"multi": Multi-choice question, where each choice contains a combination of multiple suboptions.
exp : Expert's explanation of the answer
subject_name : Medical Subject name of the particular question
topic_name : Medical topic name from the particular subject


<br/>
<br/>



# 7. Conclusion
<img width="100%" alt="Final" src="https://github.com/user-attachments/assets/2118ed95-f2fb-4a2b-aa72-9676db16dc8d">
<img width="100%" alt="Setting" src="https://github.com/user-attachments/assets/fdb8f954-ca62-4fd7-bcf7-0b53d4c436ae">
<img width="100%" alt="Running" src="https://github.com/user-attachments/assets/b7b9c204-421a-4c02-8f77-95c14be9f64a">

# 8. Things could be improved
- **Improved LLMs**:
  - Currently, I am using Llama3.2 1B model for fast response. Also, I was unable to work with vLLMs, In the future, using various tools, so that I could use stronger LLM models for better answers
  
- **LLM as a judge**:
    - Having more strong judge LLM for better understanding the chatbot generated responses
    - prompt engineering for judge LLM

- **UI/UX enhancement**:
    - This project mainly focuses on LLM and chatbot functionality. Better UI is needed in the future
