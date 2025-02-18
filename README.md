# Urbanism Annotation Project

## Overview
This repo is part of a **university-supervised research project** on automatic text annotation in the field of urbanism.  
It leverages **Ollama** and **Mistral/Llama3/Deepseek** to process input text and detect urban dynamics based on predefined annotation rules.

The model is configured using a **Modelfile** located in `prompt-playground/config/`, which defines the system prompt and behavior for urbanism-focused annotation.

---

## Installation

### 1. Clone the Repository
```bash
git clone git@github.com:valentinefleith/projet-tuteure.git
cd projet-tuteure
```
### 2. Install Dependencies

```bash
make setup
```

or just

```bash
make
```

### 3. Install Ollama and required model
Before running the project, install **Ollama** and the necessary LLM model. For example:
```bash
ollama install mistral
```
The model behavior is defined in `prompt-playground/config/Modelfile`. This file controls the **system prompt** and how the model processes urbanism-related texts.

To build the model, type:
```bash
ollama create urbaniste -f ./prompt-playground/config/Modelfile
```

If you want to use another than Mistral, you can update it in the `Modelfile` by changing the `FROM` line:
```
FROM <your-new-model>
```
Don't forget to also update it on top of `main.py` :
```py
MODEL = "mistral"
```

### 4. Run the project
Once the installation is complete, run:

```bash
make run
```
This executes `main.py` which:
- Loads the urbanism-focused LLM (`urbaniste`)
- Processes text and **classifies urban dynamics**
- Saves LLM annotation in `annotations/annotations_llm/{model-name}`
- Evaluates the model and saves results in `results/classification/{model-name}`

---
### TODO:
- Configure model dynamically
- Improve annotation accuracy by refining prompts
- Support multiple annotation types beyond urban dynamics
