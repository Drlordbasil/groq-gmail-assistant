https://youtu.be/TpBBstLW2uU?si=jdmzCQZZhOPiReVK
# Groq Gmail Assistant

The Groq Gmail Assistant is an AI-driven tool that helps manage your Gmail account by reading and responding to emails automatically. It leverages natural language processing and custom AI models to understand and generate email responses.

## Features

- **Automatic Email Processing**: Fetches and processes emails from your Gmail account.
- **AI-Driven Responses**: Uses NLTK and custom AI models for understanding content and generating replies.
- **Email Interaction**: Can read from and send emails directly through Gmail using IMAP and SMTP.
- **Automated Appointment Creation for local windows calendar app or .ics file that gets created**: This creates an .isc file and opens it for you to confirm it on your local calendar.
## Prerequisites

- Python 3.11 or higher
- IMAP access enabled in your Gmail settings
- google app password
- Install NLTK data: Run `nltk.download('stopwords')` and `nltk.download('punkt')`
- groq api key
- tbd

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Drlordbasil/groq-gmail-assistant.git
   cd groq-gmail-assistant
   ```
**TBD**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Gmail**
   - Go to account security section, create new app password after adding 2fa, email my assistant if you need help drlordbasil@gmail.com

## Configuration

1. **Email Settings**:
   - Enable IMAP in Gmail settings.
   - Allow "Less Secure Apps" if not using OAuth 2.0 (not recommended).

2. **Environment Variables**:
   Set the following environment variables:
   ```bash
   export EMAIL_USER='your-gmail-address@gmail.com'
   export EMAIL_PASS='your-app-password'  # This is the 16-character app password generated from Google's security settings
   ```
MAKE SURE TO ALSO CHANGE THE CONFIG WITHIN MAIN.PY! You can change entire system or just name/assistant name!
## Usage

1. **Running the Script**:
   ```bash
   python emailchaos.py
   ```

2. **Operation**:
   - The script will automatically connect to your Gmail via IMAP.
   - Fetches new emails and uses the LangChain AI to generate and send responses.

## Contact

For any queries or issues, reach out via email: `drlordbasil@gmail.com`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This README clearly states that an App Password should be used, which is a 16-character code that gives less secure apps or devices permission to access your Google Account. This method is recommended over using your main Gmail password, especially if two-factor authentication (2FA) is enabled. This setup helps keep your primary password secure while allowing programmable scripts like your Gmail Assistant the necessary access. Adjust the contents further if needed to better fit your project's requirements or to add additional details.Simple is best in this, easy fast groq API and gmail app password allows us to easily respond to anyone that emails. If it's live, you can email me at drlordbasil@gmail.com and ask for Chaos!
