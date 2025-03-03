from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from config.config import default_path
import json
import os
import copy

class SoT:
    def __init__(self):
        # Load the model from HF
        self.__MODEL_PATH = "saytes/SoT_DistilBERT"
        self.model = DistilBertForSequenceClassification.from_pretrained(self.__MODEL_PATH, token="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")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.__MODEL_PATH, token="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")
        
        # Load the label mapping
        self.__LABEL_MAPPING_PATH = os.path.join(str(default_path()), "config/label_mapping.json")
        self.__LABEL_MAPPING = json.load(open(self.__LABEL_MAPPING_PATH))

        # Handle paths
        self.__PROMPT_PATH_BASE = os.path.join(str(default_path()), "config/prompts/")
        self.__CONTEXT_PATH_BASE = os.path.join(str(default_path()), "config/exemplars.json")
        self.__PROMPT_FILENAMES = {
            "chunked_symbolism": "ChunkedSymbolism_SystemPrompt.md",
            "expert_lexicons": "ExpertLexicons_SystemPrompt.md",
            "conceptual_chaining": "ConceptualChaining_SystemPrompt.md",
        }

        # Handle data
        self.PROMPT_CACHE = {}
        self.CONTEXT_CACHE = {}
        
        # Preload prompts and contexts
        self.__preload_contexts()
        self.__LANGUAGE_CODES = list(self.CONTEXT_CACHE.keys())
        self.__preload_prompts()
    
    def __preload_prompts(self):
        """
        Loads all available system prompts into memory at startup.
        """

        for lang in self.__LANGUAGE_CODES:
            self.PROMPT_CACHE[lang] = {}
            for paradigm, filename in self.__PROMPT_FILENAMES.items():
                file_path = os.path.join(self.__PROMPT_PATH_BASE, lang, f"{lang}_{filename}")
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as file:
                        self.PROMPT_CACHE[lang][paradigm] = file.read()

    def __preload_contexts(self):
        """
        Loads all available contexts into memory at startup.
        """
        with open(self.__CONTEXT_PATH_BASE, "r") as f:
            self.CONTEXT_CACHE = json.load(f)
    
    def avalilable_languages(self):
        """
        Lists all currently supported languages.
        """
        return self.__LANGUAGE_CODES

    def avaliable_paradigms(self):
        """
        Returns list of all currently supported paradigms.
        """
        return list(self.__PROMPT_FILENAMES.keys())

    def get_system_prompt(self, paradigm="chunked_symbolism", language_code="EN"):
        """
        Retrieves the preloaded system prompt based on the given paradigm and language code.
        
        :param paradigm: The type of prompt (e.g., "chunked_symbolism", "expert_lexicons", "conceptual_chaining").
        :param language_code: The language code (e.g., "EN" for English, "KR" for Korean, etc.).
        :return: The content of the corresponding prompt file or None if not found.
        """
        assert paradigm in self.avaliable_paradigms(), f"`{paradigm}` is not a recognized paradigm!"
        assert language_code in self.avalilable_languages(), f"`{language_code}` is not a compatible language!"
        
        return copy.deepcopy(self.PROMPT_CACHE[language_code][paradigm])
    
        
    def get_initialized_context(self, paradigm, language_code="EN", include_system_prompt=True, format="llm"):
        """
        Retrieves the preloaded conversation context for the given paradigm and language.
        Dynamically inserts the user's question and system prompt.

        :param paradigm: The reasoning paradigm ("conceptual_chaining", "chunked_symbolism", "expert_lexicons", "cot").
        :param language_code: The language code (e.g., "KR" for Korean).
        :param include_system_prompt: Whether to add the system prompt to the context. Not available in raw format.
        :param format: The format to return. Accepted values are: `llm`, `raw`, or `vlm`.
        :return: The full initialized conversation list.
        """

        assert paradigm in self.avaliable_paradigms(), f"`{paradigm}` is not a recognized paradigm!"
        assert language_code in self.avalilable_languages(), f"`{language_code}` is not a compatible language!"

        if format.lower() == "llm":
            exemplars = self.CONTEXT_CACHE[language_code][paradigm]
            if include_system_prompt:
                context = [{"role": "system", "content": self.get_system_prompt(paradigm=paradigm, language_code=language_code)}]
            else:
                context = []

            for ex in exemplars:
                context += [
                    {"role": "user", "content": ex['question']},
                    {"role": "assistant", "content": ex['answer']},
                ]
            return context
        
        elif format.lower() == "vlm":
            exemplars = self.CONTEXT_CACHE[language_code][paradigm]
            if include_system_prompt:
                context = [{"role": "system", "content": [{"type": "text", "text": self.get_system_prompt(paradigm=paradigm, language_code=language_code)}]}]
            else:
                context = []

            for ex in exemplars:
                context += [
                    {"role": "user", "content": [{"type": "text", "text": ex['question']}]},
                    {"role": "assistant", "content": [{"type": "text", "text": ex['answer']}]},
                ]
            return context
        
        else:  # Default case, return raw format
            return copy.deepcopy(self.CONTEXT_CACHE[language_code][paradigm])
    
    def classify_question(self, question):
        """
        Uses the pretrained DistilBERT model to classify the paradigm of a question.
        """

        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # Reverse mapping to get the paradigm name
        label_mapping_reverse = {v: k for k, v in self.__LABEL_MAPPING.items()}
        return label_mapping_reverse[predicted_class]
