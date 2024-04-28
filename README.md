

# Groq Gmail Assistant: AI-Enhanced Email Management

**Groq Gmail Assistant** leverages advanced AI to automate and enhance email management, streamlining the process of reading and responding to emails using technologies like Groq, LangChain, and Ollama.

## Key Features
- **Automated Email Processing**: Reads and responds to emails automatically.
- **Advanced Sentiment Analysis**: Assesses the tone and intent of messages.
- **Seamless Calendar Integration**: Manages appointments directly through email interactions.
- **Enhanced Contextual Understanding**: Uses detailed note-taking and web searches to enhance response relevance.

## Getting Started

### Prerequisites
- Python (version 3.x recommended)
- Gmail account with IMAP access enabled
- Google App Password for secure access

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Drlordbasil/groq-gmail-assistant.git
   cd groq-gmail-assistant
   ```

2. **Install Dependencies**:
   Install all necessary Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Enable IMAP in Gmail**:
   - Go to your Gmail settings.
   - Under the 'Forwarding and POP/IMAP' tab, enable IMAP.

2. **Generate Google App Password**:
   - Visit the Google App Password page (https://myaccount.google.com/apppasswords).
   - Generate a new password specifically for the Gmail Assistant.

3. **Set Up Configuration File**:
   - Rename `example.config.py` to `config.py`.
   - Edit `config.py` to include your Gmail address and the newly generated App Password.

### Usage

1. **Run the Assistant**:
   Execute the script to start processing emails:
   ```bash
   python emailchaos.py
   ```

## Support

For any issues or inquiries, reach out via email at `drlordbasil@gmail.com`. Experience hands-on interaction with Chaos, your AI assistant, by sending an email!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
