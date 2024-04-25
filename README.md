![image](https://github.com/Drlordbasil/groq-gmail-assistant/assets/126736516/f2b23158-79d9-4942-a578-21c2181fa78b)

https://youtu.be/TpBBstLW2uU?si=jdmzCQZZhOPiReVK
# Groq Gmail Assistant

The Groq Gmail Assistant is an AI-driven tool that helps manage your Gmail account by reading and responding to emails automatically. It leverages natural language processing, sentiment analysis, and custom AI models to understand and generate email responses.

## Features

- **Automatic Email Processing**: Fetches and processes emails from your Gmail account.
- **AI-Driven Responses**: Uses advanced sentiment analysis models and custom AI models for understanding content and generating contextually relevant replies.
- **Email Interaction**: Can read from and send emails directly through Gmail using IMAP and SMTP.
- **Automated Appointment Creation**: Creates an .ics file for appointments and opens it for you to confirm on your local calendar application.
- **Web Search Integration**: Performs web searches using Selenium to gather relevant information for generating email responses.
- **Note-taking Capabilities**: Can write and read notes from a local file for enhanced context understanding.

## Prerequisites

- Python (latest version)
- IMAP access enabled in your Gmail settings
- Google App Password
- Install NLTK data: Run `nltk.download('stopwords')` and `nltk.download('punkt')`
- Groq API Key
- ChromeDriver for Selenium (automatically installed by the script)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Drlordbasil/groq-gmail-assistant.git
   cd groq-gmail-assistant
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Gmail**:
   - Go to the account security section in your Gmail settings.
   - Enable 2-factor authentication (2FA) if not already enabled.
   - Generate a new App Password for the Gmail Assistant.
   - If you need help, feel free to email the assistant at drlordbasil@gmail.com.

## Configuration

1. **Email Settings**:
   - Enable IMAP in your Gmail settings.
   - Update the `config.py` file with your Gmail email address and the generated App Password.

2. **Customization**:
   - You can customize the entire system or just the name/assistant name in the `config.py` file.
   - Modify the `SYSTEM_PROMPT` variable to adjust the assistant's behavior and personality.

## Usage

1. **Running the Script**:

   ```bash
   python emailchaos.py
   ```

2. **Operation**:
   - The script will automatically connect to your Gmail via IMAP.
   - It fetches new emails and uses the LangChain Ollama-based AI to generate and send responses by passing them to the Groq-based AI.

## Contact

For any queries or issues, reach out via email: `drlordbasil@gmail.com`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

If the Gmail Assistant is live, you can email drlordbasil@gmail.com and ask for Chaos, the AI assistant, to see it in action!
