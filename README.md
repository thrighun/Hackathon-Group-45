# 🚀 Self-Learning Job Description Enhancement System

An AI-powered system that transforms raw, unstructured job descriptions into **structured, professional, and ATS-friendly formats**, while continuously improving through user feedback and iterative fine-tuning.

---

## 📌 Overview

Job descriptions are often inconsistent, messy, and time-consuming to standardize. This project solves that by building a **self-hosted AI pipeline** that:

- Converts raw job descriptions into structured JSON
- Enhances clarity, grammar, and professionalism
- Learns from user edits over time
- Reduces dependency on large external LLM APIs

This system is designed as part of the **Content Structuring and Enhancement Engine** problem statement.

---

## 🎯 Key Features

- 🧠 **Custom AI Model (<4B params)** using Hugging Face Transformers  
- ⚡ **Fast Inference API** with FastAPI / Flask  
- 🔄 **Self-Learning Pipeline**
  - Stores user edits  
  - Retrains periodically  
- 📊 **Evaluation Metrics**
  - Perplexity  
  - ROUGE / BLEU  
  - Structural completeness  
- 💾 **Database Integration**
  - PostgreSQL / MongoDB  
- 🌐 **Web / CLI Interface (Optional)**  

---

## 🏗️ System Architecture
Raw JD Input
↓
Preprocessing
↓
Fine-tuned Model (LoRA / PEFT)
↓
Structured JSON Output
↓
User Edits (Feedback Loop)
↓
Database Storage
↓
Periodic Fine-tuning
↓
Improved Model


---

## 🧠 How It Works

1. User inputs a raw job description  
2. Model generates structured output  
3. User optionally edits the result  
4. Edited output is stored  
5. System periodically retrains using new data  
6. Model improves over time  

---

## 🛠️ Tech Stack

 Layer        | Technology                     
--------------|-------------------------------
 Backend      | FastAPI / Flask               
 Model        | Hugging Face Transformers     
 Training     | PyTorch + PEFT (LoRA / QLoRA) 
 Database     | MongoDB                       

Model Training
Base model: gemma-2b-it
Fine-tuning: LoRA / QLoRA
Dataset: Raw → Enhanced JD pairs
