# Sketch-of-Thought (SoT)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Compatible-yellow)](https://huggingface.co/saytes/SoT_DistilBERT)

## Introduction

Sketch-of-Thought (SoT) is a novel prompting framework for efficient reasoning in language models that combines cognitive-inspired reasoning paradigms with linguistic constraints to minimize output token usage while preserving reasoning accuracy.

Unlike conventional Chain of Thought (CoT) approaches that produce verbose reasoning chains, SoT implements three distinct reasoning paradigms:

- **Conceptual Chaining**: Connects essential ideas in logical sequences through structured step links. Effective for commonsense reasoning, multi-hop inference, and fact-based recall tasks.
  
- **Chunked Symbolism**: Organizes numerical and symbolic reasoning into structured steps with equations, variables, and arithmetic operations. Excels in mathematical problems and technical calculations.
  
- **Expert Lexicons**: Leverages domain-specific shorthand, technical symbols, and jargon for precise and efficient communication. Suited for technical disciplines requiring maximum information density.

SoT includes a paradigm selection model that automatically determines the optimal reasoning approach for a given query, eliminating the need for manual heuristics.

## System Prompts

Here are the system prompts used in our paper. We offer them in English, Korean, Italian, and German.

| Language | Available System Prompts |
|----------|------------------------------|
| English (EN) | • [Chunked Symbolism](sketch_of_thought/config/prompts/EN/EN_ChunkedSymbolism_SystemPrompt.md)<br>• [Conceptual Chaining](sketch_of_thought/config/prompts/EN/EN_ConceptualChaining_SystemPrompt.md)<br>• [Expert Lexicons](sketch_of_thought/config/prompts/EN/EN_ExpertLexicons_SystemPrompt.md) |
| Korean (KR) | • [Chunked Symbolism](sketch_of_thought/config/prompts/KR/KR_ChunkedSymbolism_SystemPrompt.md)<br>• [Conceptual Chaining](sketch_of_thought/config/prompts/KR/KR_ConceptualChaining_SystemPrompt.md)<br>• [Expert Lexicons](sketch_of_thought/config/prompts/KR/KR_ExpertLexicons_SystemPrompt.md) |
| Italian (IT) | • [Chunked Symbolism](sketch_of_thought/config/prompts/IT/IT_ChunkedSymbolism_SystemPrompt.md)<br>• [Conceptual Chaining](sketch_of_thought/config/prompts/IT/IT_ConceptualChaining_SystemPrompt.md)<br>• [Expert Lexicons](sketch_of_thought/config/prompts/IT/IT_ExpertLexicons_SystemPrompt.md) |
| German (DE) | • [Chunked Symbolism](sketch_of_thought/config/prompts/DE/DE_ChunkedSymbolism_SystemPrompt.md)<br>• [Conceptual Chaining](sketch_of_thought/config/prompts/DE/DE_ConceptualChaining_SystemPrompt.md)<br>• [Expert Lexicons](sketch_of_thought/config/prompts/DE/DE_ExpertLexicons_SystemPrompt.md) |

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/SimonAytes/SoT.git
   cd SoT
   ```

2. **Create a Conda Environment (Recommended)**

   ```bash
   conda create -n sot python=3.10 -y
   conda activate sot
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Quickstart

Here's a minimal example showing how to use SoT with any LLM:

```python
from sketch_of_thought import SoT

# Initialize SoT
sot = SoT()

# Classify a question to determine the best reasoning paradigm
question = "Alice has 5 apples. She gives 3 apples to Bob. How many apples does Alice have?"
paradigm = sot.classify_question(question)
# Returns: 'chunked_symbolism'

# Get the appropriate system prompt for the paradigm
system_prompt = sot.get_system_prompt(paradigm)

# Get initialized context with exemplars for the selected paradigm
context = sot.get_initialized_context(
    paradigm=paradigm, 
    question=question, 
    format="llm",
    include_system_prompt=True
)

# The context can now be passed to any LLM
```

## Example with Qwen2.5-7B

Here's a complete example using Qwen2.5-7B-Instruct:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sketch_of_thought import SoT

# Initialize SoT
sot = SoT()

# Load Qwen model
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare the question
prompt = "Alice has 5 apples. She gives 3 apples to Bob. How many apples does Alice have?"

# Classify and get appropriate context
paradigm = sot.classify_question(prompt)
messages = sot.get_initialized_context(
    paradigm,
    prompt,
    format="llm",
    include_system_prompt=True
)

# Format for the model
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate response
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode response
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

**Output:**

```
<think>
A = 5
A -= 3
A = 2
</think>

\boxed{2}
```

## Helper Functions

SoT provides several utility functions:

```python
# List available reasoning paradigms
sot.avaliable_paradigms()
# Returns: ['chunked_symbolism', 'conceptual_chaining', 'expert_lexicons']

# List supported languages
sot.avalilable_languages()
# Returns: ['EN', 'KR', 'IT', 'DE']

# Get formatted context without a question
context = sot.get_initialized_context(paradigm="conceptual_chaining", format="llm")

# Get raw exemplars
raw_examples = sot.get_initialized_context(paradigm="chunked_symbolism", format="raw")
```

## Supported Formats

Our code supports multiple output formats:

- `"llm"`: Standard chat format for text-only LLMs
- `"vlm"`: Multimodal format for vision-language models
- `"raw"`: Raw exemplars without formatting

<details>
  <summary>What's the difference?</summary>
  
  ### LLM Format

  Standard `messages` format for Large Language Models.

  ```python
  [
    {
      "role": "system", 
      "content": "SYSTEM_PROMPT_HERE"
    },
    {
      "role": "user", 
      "content": "EXAMPLE_QUESTION_HERE"
    },
    {
      "role": "assistant", 
      "content": "EXAMPLE_ANSWER_HERE"
    },
    {
      "role": "user", 
      "content": "USER_QUESTION_HERE"
    }
  ]
  ```
  
  ### VLM Format

  Standard `messages` format for Large Vision-Language Models.
  
  ```python
  [
    {
      "role": "system", 
      "content": "SYSTEM_PROMPT_HERE"
    },
    {
      "role": "user", 
      "content": [{"type": "text", "text": "EXAMPLE_QUESTION_HERE"}]
    },
    {
      "role": "assistant", 
      "content": [{"type": "text", "text": "EXAMPLE_ANSWER_HERE"}]
    },
    {
      "role": "user", 
      "content": [{"type": "text", "text": "USER_QUESTION_HERE"}]
    }
  ]
  ```
  
  ### Raw Format

  Raw exemplar data. Apply your own format!

  ```python
  [
    {
      "question": "EXAMPLE_QUESTION_HERE",
      "answer": "EXAMPLE_ANSWER_HERE"
    },
    {
      "question": "EXAMPLE_QUESTION_HERE",
      "answer": "EXAMPLE_ANSWER_HERE"
    }
  ]
  ```
</details>

## Multilingual Support

SoT supports multiple languages (depending on your configuration). System prompts and exemplars are automatically loaded in the requested language.

## Paradigm Selection Model

SoT includes a pretrained DistilBERT model for automatic paradigm selection based on the question. The model is available on Hugging Face: [saytes/SoT_DistilBERT](https://huggingface.co/saytes/SoT_DistilBERT)

## Datasets

The SoT_DistilBERT model was evaluated on the following datasets:

| Dataset | HF ID | Subset | Split | Evaluation Type |
|---------|-------|--------|-------|----------------|
| GSM8K | [gsm8k](https://huggingface.co/datasets/gsm8k) | main | test | numerical |
| SVAMP | [ChilleD/SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) | - | test | numerical |
| AQUA-RAT | [aqua_rat](https://huggingface.co/datasets/aqua_rat) | - | test | multiple_choice |
| DROP | [drop](https://huggingface.co/datasets/drop) | - | validation | open |
| OpenbookQA | [openbookqa](https://huggingface.co/datasets/openbookqa) | - | test | multiple_choice |
| StrategyQA | [ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA) | - | test | yesno |
| LogiQA | [lucasmccabe/logiqa](https://huggingface.co/datasets/lucasmccabe/logiqa) | default | test | multiple_choice |
| Reclor | [metaeval/reclor](https://huggingface.co/datasets/metaeval/reclor) | - | validation | multiple_choice |
| HotPotQA | [hotpot_qa](https://huggingface.co/datasets/hotpot_qa) | distractor | validation | open |
| MuSiQue-Ans | [dgslibisey/MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) | - | validation | open |
| QASC | [allenai/qasc](https://huggingface.co/datasets/allenai/qasc) | - | validation | multiple_choice |
| Worldtree | [nguyen-brat/worldtree](https://huggingface.co/datasets/nguyen-brat/worldtree) | - | train | multiple_choice |
| PubMedQA | [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | pqa_labeled | train | yesno |
| MedQA | [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa) | med_qa_en_source | validation | multiple_choice |

## Citation

If you find our work helpful, please cite:

```
@misc{aytes2025sot,
      title={Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching}, 
      author={Simon A. Aytes and Jinheon Baek and Sung Ju Hwang},
      year={2025},
      eprint={2503.05179},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.05179}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
