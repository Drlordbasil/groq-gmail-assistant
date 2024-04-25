![image](https://github.com/Drlordbasil/groq-gmail-assistant/assets/126736516/f2b23158-79d9-4942-a578-21c2181fa78b)

https://youtu.be/TpBBstLW2uU?si=jdmzCQZZhOPiReVK
# Groq Gmail Assistant: AI-Powered Email Management

The Groq Gmail Assistant is an AI-driven tool that helps manage your Gmail account by automatically reading and responding to emails. It utilizes natural language processing, sentiment analysis, and custom AI models to understand email content and generate human-like responses. With features like automated email processing, AI-driven responses, appointment creation, web search integration, and note-taking capabilities, the Groq Gmail Assistant streamlines your email management workflow.

## How Chaos Works: A Step-by-Step Process

1. **Email Fetching**: Chaos connects to your Gmail account using IMAP and fetches new, unread emails.

2. **Email Parsing**: The fetched emails are parsed to extract relevant information, such as the sender, subject, and body content.

3. **Text Preprocessing**: The email body undergoes text preprocessing techniques, including removing stop words, tokenization, and cleaning up HTML content.

4. **Sentiment Analysis**: Advanced sentiment analysis models are applied to the preprocessed email content to determine the overall sentiment and tone of the email.

5. **Relevance Determination**: Chaos analyzes the email content to determine if a response is required based on predefined criteria and the results of the sentiment analysis.

6. **Response Generation**: If a response is deemed necessary, Chaos uses the LangChain Ollama-based AI to generate a contextually relevant response by considering the email content, sentiment, and any relevant notes or web search results.

7. **Tool Utilization**: Chaos can leverage various tools to enhance the response generation process, such as creating calendar appointments, performing web searches, writing notes, or reading previously saved notes.

8. **Response Refinement**: The generated response is further refined and optimized using the Groq-based AI to ensure coherence, clarity, and appropriateness.

9. **Email Sending**: The final response is sent back to the original sender using SMTP, effectively automating the email reply process.

10. **Logging and Monitoring**: Throughout the process, Chaos logs important events and actions for monitoring and debugging purposes.

By automating this end-to-end email management process, Chaos saves you time and effort while providing timely and relevant responses to your emails.

## Key Features

- 🚀 Automated email processing and response generation
- 🧠 Advanced sentiment analysis for understanding email tone
- 📅 Seamless appointment creation and calendar integration
- 🔍 Web search capabilities for informed decision-making
- 📝 Note-taking functionality for enhanced context understanding
- ⚙️ Customizable system prompts and assistant personality
- 🔒 Secure authentication using Google App Passwords
- 📧 Compatible with the latest Python version and Gmail IMAP/SMTP protocols

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

## Getting Started

To start using the Groq Gmail Assistant, follow the installation and configuration steps outlined in the README. If you encounter any issues or have questions, feel free to reach out to drlordbasil@gmail.com for assistance.

## Unlock the Power of AI in Your Email Management

Say goodbye to manual email management and let Chaos, the AI-powered Gmail assistant, handle your emails with intelligence and efficiency. With its advanced natural language processing capabilities and seamless integration with Gmail, Chaos revolutionizes the way you interact with your inbox.

Try the Groq Gmail Assistant today and experience the future of email management! 🌟

---

#AI #EmailManagement #Automation #ProductivityHack #NLP #SentimentAnalysis #Groq #LangChain #Ollama #Python #Gmail #IMAP #SMTP

## Contact

For any queries or issues, reach out via email: `drlordbasil@gmail.com`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

If the Gmail Assistant is live, you can email drlordbasil@gmail.com and ask for Chaos, the AI assistant, to see it in action!
