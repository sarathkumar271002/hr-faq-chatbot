# HR FAQ Chatbot

A Flask-based chatbot that answers HR-related FAQs using semantic search with sentence-transformers. Optionally, it integrates with OpenAI models for more natural responses if an API key is provided.

## Features

* Search FAQs using semantic embeddings (`all-MiniLM-L6-v2`).
* Fallback to direct FAQ answers without LLM if OpenAI API key is not set.
* REST API endpoints for asking questions and reindexing FAQs.
* Caching of embeddings for faster startup.
* Easily extendable by updating the `faqs.csv` file.

## Project Structure

```
.
├── app.py              # Main Flask app
├── config.py           # Configuration for OpenAI API key and model
├── faqs.csv            # FAQ dataset (questions and answers)
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend (if added)
└── faq_index.pkl       # Cached embeddings (auto-generated)
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/hr-faq-chatbot.git
   cd hr-faq-chatbot
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Configure OpenAI:

   * Copy `.env.example` to `.env`
   * Add your OpenAI API key:

     ```
     OPENAI_API_KEY=sk-xxxx
     OPENAI_MODEL=gpt-4o-mini
     ```

## Usage

1. Run the app:

   ```bash
   python app.py
   ```

   The app will start on `http://0.0.0.0:5000`.

2. API Endpoints:

   * `POST /ask`
     Request body:

     ```json
     {
       "query": "What is the leave policy?"
     }
     ```

     Response:

     ```json
     {
       "answer": "Employees are entitled to ...",
       "matches": [
         {"question": "...", "answer": "...", "score": 0.87}
       ]
     }
     ```

   * `POST /reindex`
     Rebuild embeddings if `faqs.csv` is updated.

3. Updating FAQs:

   * Add or modify entries in `faqs.csv`.
   * Call `/reindex` endpoint or restart the app.

## Dependencies

See `requirements.txt`:

* Flask
* pandas
* numpy
* sentence-transformers
* torch
* openai (optional)


