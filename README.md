# 🚀 IGL Document Intelligence Chatbot

## Overview
A high-performance, quantized document chatbot powered by Mistral:7b-instruct-v0.2-q4_K_M and designed for efficient document querying with minimal computational overhead.

## Features
- 📄 Advanced document embedding and retrieval
- 💬 Streamlit-based interactive interface
- 🔍 Semantic search capabilities
- 💾 Efficient memory management

## Prerequisites

### 1. System Requirements
- Windows 10/11 or Linux/macOS
- Python 3.8+
- CUDA-capable GPU (recommended, but not mandatory)

### 2. Install Ollama (Windows)

#### Step-by-Step Ollama Installation:
1. Download Ollama for Windows:
   - Visit [Ollama Official Website](https://ollama.com/download/windows)
   - Download the installer
   - Run the downloaded `.exe` file

2. Verify Ollama Installation:
   ```bash
   ollama --version
   ```

### 3. Start Ollama Service
```bash
ollama serve
```

### 4. Pull Quantized Mistral Model
```bash
ollama pull mistral:7b-instruct-v0.2-q4_K_M
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/igl-document-chatbot.git
cd igl-document-chatbot
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Prepare Your Document
- Place your document (`IGL FAQs (1).docx`) in the project directory
- Ensure the filename matches in the code

## Running the Chatbot
```bash
streamlit run document_chatbot.py
```

## Troubleshooting
- Ensure Ollama is running before starting the chatbot
- Check CUDA compatibility if using GPU acceleration
- Verify Python and pip versions

## Performance Tips
- First run might be slower due to model loading
- Subsequent interactions will be faster
- Allocate more RAM for better performance

