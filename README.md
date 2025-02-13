# St. Modwen Home Specialist: LLM-Powered Recommendation System

The St. Modwen Home Specialist is an LLM-powered recommendation engine designed to recommend homs in different development, based on the user's preferences. Leveraging state-of-the-art language models, semantic search, and vector database, the system processes data from various sources and provides detailed, context-rich recommendations based on user queries.



![Chesterfield](Images/Chest.jpg)
![Wales](Images/wales.jpg)


## Features
* LLM-Driven Recommendations: It Utilizes GPT-4 and GPT-3.5-turbo to interpret user queries, rephrase their queries and follow-ups, and generate detailed recommendations for homes.

* Data Ingestion: Downloads and processes data (CSV, JSON, HTML) from remote sources using API endpoints. The system automatically converts CSV data to JSON format and preprocesses it for streamlined processing.

* Semantic Search & FAQ Integration: Builds a vector databse using Chroma to create embeddings for both site content and FAQs to form the knowledge base, ensuring that similar queries are matched to the most relevant information.

Interactive Chat Interface: Employs Streamlit to provide an interactive, chat-based user interface. The system maintains chat history by syncing with the remote Ikigai API, ensuring continuity in conversations.


## Usage

### 1. Clone this repository to your local machine.
Start by cloning the GitHub repository containing the Telco Churn Analysis project and the predictive app. You can do this by running the following command in your terminal:
```
https://github.com/KimathiNewton/realestate-chat-assistant.git
```
Navigate to the project directory.
### 2. Setup Virtual Environment
You need Python3 on your system to setup this app. Then you can clone this repo and being at the repo's root :: streamlit sales prediction app> ... follow the steps below:

Windows
```
    python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt 
```
Linux & MacOs
```
    python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
```
### 3. Install Dependencies:
Install the required Python packages within your virtual environment:
```
pip install -r requirements.txt
```
### 4. Set up environment variables:
Create a .env file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key_here

```
To run the Application Script

```
python app.py
```

