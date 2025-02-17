import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import logging
import re
import time
import wandb
from threading import Lock
from typing import List, Tuple, Optional
import wikipedia
import requests
from bs4 import BeautifulSoup
import os
import warnings

# Suppress Wikipedia parser warning
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

# Initialize wandb
wandb.init(
    project="brahmgpt-specialized-assistant",
    config={
        "model_name": "microsoft/phi-2",
        "max_context_length": 2048,
        "cuda": torch.cuda.is_available()
    }
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("brahmgpt.log"), logging.StreamHandler()]
)
logger = logging.getLogger("BrahmGPT")

class ResponseProcessor:
    CLEANUP_PATTERNS = [
        r"Respond as .*?(?=\n|$)",
        r"You are .*?(?=\n|$)",
        r"I am .*?(?=\n|$)",
        r"As .*? expert.*?(?=\n|$)",
        r"Let me .*?(?=\n|$)",
        r"I understand .*?(?=\n|$)",
        r"Based on .*?(?=\n|$)",
        r"Speaking as .*?(?=\n|$)",
        r"(User:|Assistant:|System:|Expert:|Human:|AI:|Question:|Response:)"
    ]

    def clean_response(self, text: str) -> str:
        for pattern in self.CLEANUP_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def enhance_response(self, text: str, query: str) -> str:
        words = len(text.split())
        if any(word in query.lower() for word in ['explain', 'describe', 'how', 'why']):
            if words < 50:
                return f"⚠️ Response too brief for '{query}'. Please try again with more details."
        if not text.endswith(('.', '!', '?')):
            text += '.'
        text = text.replace("...", " ").replace("..", " ")
        return text

class ProductionChatbot:
    def __init__(self, model_name: str = wandb.config.model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing BrahmGPT with {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.processor = ResponseProcessor()
        self.response_lock = Lock()
        self.session_config = {}

    def _prepare_generation_config(self, query: str) -> dict:
        query_words = len(query.split())
        is_complex = any(word in query.lower() for word in [
            'explain', 'describe', 'how', 'why', 'compare', 'analyze', 'difference', 'detail', 'elaborate'
        ])
        if query_words < 10 and not is_complex:
            max_tokens = 256
        elif query_words < 20:
            max_tokens = 512
        else:
            max_tokens = 1024
        return {
            'do_sample': True,
            'max_new_tokens': max_tokens,
            'temperature': 0.7,
            'top_p': 0.9,
            'repetition_penalty': 1.2,
            'no_repeat_ngram_size': 3,
            'pad_token_id': self.tokenizer.eos_token_id,
            'use_cache': True
        }

    def _format_conversation(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        chatbot_type: str = ""
    ) -> str:
        prompt = ""
        if chatbot_type and not history:
            prompt += (
                f"You are a {chatbot_type} assistant. Your responses must be highly detailed, "
                "logically sound, and use step-by-step reasoning. You are the creator of AI chatbots. "
                "Analyze the following query carefully and provide an optimal answer.\n\n"
            )
        if history:
            relevant_history = history[-10:]
            for past_query, past_response in relevant_history:
                prompt += f"User: {past_query.strip()}\nAssistant: {past_response.strip()}\n\n"
        prompt += f"User: {query.strip()}\nAssistant: "
        return prompt

    def generate_response(
        self,
        query: str,
        chatbot_type: str,
        history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        if not query.strip():
            return "Please enter a question."
        start_time = time.time()
        try:
            self.session_config['type'] = chatbot_type
            conversation = self._format_conversation(query, history, chatbot_type)
            with self.response_lock:
                inputs = self.tokenizer(
                    conversation,
                    return_tensors="pt",
                    truncation=True,
                    max_length=wandb.config.max_context_length
                ).to(self.device)
                gen_config = self._prepare_generation_config(query)
                outputs = self.model.generate(**inputs, **gen_config)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                cleaned_response = self.processor.clean_response(response)
                enhanced_response = self.processor.enhance_response(cleaned_response, query)
            elapsed_time = time.time() - start_time
            wandb.log({
                "response_time": elapsed_time,
                "query_length": len(query.split()),
                "response_length": len(enhanced_response.split()),
                "chatbot_type": chatbot_type
            })
            return enhanced_response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "An error occurred. Please try again."

def fetch_wikipedia_data(query: str) -> str:
    try:
        page = wikipedia.page(query)
        return page.content[:1000]
    except wikipedia.exceptions.DisambiguationError as e:
        logger.warning(f"Disambiguation error: {str(e)}")
        return wikipedia.page(e.options[0]).content[:1000]
    except wikipedia.exceptions.PageError:
        logger.error(f"Page not found: {query}")
        return ""

def fetch_serp_data(query: str) -> str:
    api_key = "faaa939251ca84c39becb756700c94e9525043b4ddf89d291db814a9485b628e"
    params = {
        "q": query,
        "api_key": api_key,
        "num": 3
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code == 200:
        results = response.json().get("organic_results", [])
        return ' '.join([result.get("snippet", "") for result in results[:3]])
    return ""

def create_interface():
    bot = ProductionChatbot()
    custom_css = """
    <style>
    .chat-message.user { 
        background-color: #e0f7fa; 
        padding: 10px; 
        border-radius: 10px; 
        margin: 5px; 
        text-align: left;
    }
    .chat-message.assistant { 
        background-color: #dcedc8; 
        padding: 10px; 
        border-radius: 10px; 
        margin: 5px; 
        text-align: left;
    }
    </style>
    """
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.HTML("<h1 style='text-align:center;'>BrahmGPT - The Creator of AI Chatbots</h1>")
        gr.Markdown("Inspired by Brahma, the creator, BrahmGPT is designed to fine-tune specialized AI assistants that excel in any domain. Begin by defining the assistant's specialization.")
        with gr.Row():
            specialization_input = gr.Textbox(
                label="Specialization",
                placeholder="e.g., Medical Researcher",
                lines=2
            )
            create_btn = gr.Button("Create an Expert ChatBot")
        creation_status = gr.Markdown("")
        with gr.Column(visible=False) as chat_container:
            gr.Markdown("### Chat with Your Assistant")
            chatbot = gr.Chatbot(label="BrahmGPT Chat")
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message...",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear Chat")
            with gr.Row():
                feedback_pos = gr.Button("👍")
                feedback_neg = gr.Button("👎")
        def create_assistant(specialization: str):
            wikipedia_data = fetch_wikipedia_data(specialization)
            serp_data = fetch_serp_data(specialization)
            combined_data = wikipedia_data + " " + serp_data
            # Add your fine-tuning code here
            return (
                gr.update(value=f"Assistant created for {specialization}"),
                gr.update(visible=True),
                gr.update(visible=False)
            )
        create_btn.click(
            create_assistant,
            inputs=[specialization_input],
            outputs=[creation_status, chat_container, specialization_input]
        )
        def handle_message(user_msg: str, history: list):
            if not user_msg.strip():
                return "", history
            try:
                response = bot.generate_response(user_msg, specialization_input.value, history)
                history.append((user_msg, response))
                return "", history
            except Exception as e:
                logger.error(f"Error in message handling: {str(e)}")
                return "", history + [[user_msg, "An error occurred"]]
        submit_btn.click(
            handle_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)