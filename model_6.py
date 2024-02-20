import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, MarianMTModel, MarianTokenizer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from IPython.display import display

class Chatbot:
    def __init__(self, translation_model_name, source_language_code, target_language_code, model, tokenizer, retriever, llm, chat_history):
        self.translation_model = MarianMTModel.from_pretrained(translation_model_name)
        self.translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
        self.source_language_code = source_language_code
        self.target_language_code = target_language_code

        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.llm = llm
        self.chat_history = chat_history

    def translate(self, text):
        inputs = self.translation_tokenizer(text, return_tensors="pt", truncation=True)
        translation = self.translation_model.generate(**inputs)
        translated_text = self.translation_tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
        return translated_text

    def create_conversation(self, query):
        try:
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=False
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=memory,
                get_chat_history=lambda h: h,
            )

            result = qa_chain({'question': query, 'chat_history': self.chat_history})
            self.chat_history.append((query, result['answer']))
            return '', self.chat_history

        except Exception as e:
            self.chat_history.append((query, e))
            return '', self.chat_history

    def on_submit_button_click(self, b):
        query = self.query_input.value
        use_spanish = self.spanish_checkbox.value

        if use_spanish:
            translated_query = self.translate(query)
        else:
            translated_query = query

        response, self.chat_history = self.create_conversation(translated_query)

        if use_spanish:
            translated_response = self.translate(response)
        else:
            translated_response = response

        with self.response_output:
            print(translated_response)

        self.chat_history_output.clear_output(wait=True)
        with self.chat_history_output:
            for entry in self.chat_history:
                print(f"User: {entry[0]}")
                print(f"Bot: {entry[1]}")

    def display_widgets(self):
        self.query_input = widgets.Text(description="Your message:")
        self.response_output = widgets.Output()
        self.chat_history_output = widgets.Output()

        self.spanish_checkbox = widgets.Checkbox(value=False, description="Use Spanish")

        self.submit_button = widgets.Button(description="Send")
        self.submit_button.on_click(self.on_submit_button_click)

        display(self.query_input, self.spanish_checkbox, self.submit_button, self.response_output, self.chat_history_output)

class ChatbotTraining:
    def __init__(self, translation_model_name, source_language_code, target_language_code, model_name, folder_path, embedding_model_name):
        self.translation_model = MarianMTModel.from_pretrained(translation_model_name)
        self.translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
        self.source_language_code = source_language_code
        self.target_language_code = target_language_code

        self.model = self.load_quantized_model(model_name)
        self.tokenizer = self.initialize_tokenizer(model_name)

        drive.mount('/content/drive')
        self.folder_path = folder_path
        self.documents = self.load_documents()

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.all_splits = self.text_splitter.split_documents(self.documents)

        self.embedding_model_name = embedding_model_name
        self.model_kwargs = {"device": "cuda"}
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name, model_kwargs=self.model_kwargs)

        self.vectordb = Chroma.from_documents(documents=self.all_splits, embedding=self.embeddings, persist_directory="chroma_db")
        self.retriever = self.vectordb.as_retriever()

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2048,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

    def load_quantized_model(self, model_name):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        return model

    def initialize_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.bos_token_id = 1
        return tokenizer

    def load_documents(self):
        loader = PyPDFDirectoryLoader(self.folder_path)
        return loader.load()

# Usage Example:
translation_model_name = "path/to/your/translation/model"
source_language_code = "es"
target_language_code = "en"
model_name = "path/to/your/model"
folder_path = '/content/drive/MyDrive/chatbot_data/'
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

chatbot_training = ChatbotTraining(translation_model_name, source_language_code, target_language_code, model_name, folder_path, embedding_model_name)

# Create chatbot instances
chatbot_1 = Chatbot(translation_model_name, source_language_code, target_language_code,
                   chatbot_training.model, chatbot_training.tokenizer,
                   chatbot_training.retriever, chatbot_training.llm, [])

chatbot_2 = Chatbot(translation_model_name, source_language_code, target_language_code,
                   chatbot_training.model, chatbot_training.tokenizer,
                   chatbot_training.retriever, chatbot_training.llm, [])

# Display widgets for chatbots
chatbot_1.display_widgets()

chatbot_2.display_widgets()

"""An ITAS data portal is a web application that offers the extensive functionality of the trader's desk and blends the extensive experience of the trading industry with the most up-to-date technology and an improved user experience. It allows you to access information about business entities and companies, display active information in activity tables, and provides customizable data views."""