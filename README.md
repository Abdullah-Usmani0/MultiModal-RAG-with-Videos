# 🎬 MultiModal RAG with Videos

**MultiModal RAG with Videos** is an advanced, AI-powered application that enables users to extract and interact with both visual and textual content from YouTube videos. By integrating state-of-the-art language models (LLMs), this tool allows users to ask complex questions about a video’s content, using a Retrieval-Augmented Generation (RAG) approach. The app processes video, audio, and frames to provide context-aware, accurate responses.

🚀 Features
- **Download and Process YouTube Videos**: Automatically download YouTube videos, extract key frames, and transcribe audio.
- **MultiModal Indexing**: Use both visual (frames) and textual (transcribed audio) data to create a multi-modal vector store, enhancing query accuracy.
- **Real-time Question Answering**: Ask questions about the content of the video and receive detailed answers using OpenAI’s GPT-4-turbo multimodal model.
- **Customizable Video Queries**: Users can input any YouTube video and ask questions about its specific sections, making it ideal for deep dives into educational, tutorial, or any video content.
- **Streamlit Interface**: Clean and user-friendly UI built using Streamlit for seamless interaction.

### 🔧 Technologies Used
- **[OpenAI GPT-4-turbo](https://platform.openai.com/docs/models/gpt-4)**: For generating answers based on retrieved text and images.
- **[LlamaIndex](https://gpt-index.readthedocs.io/en/latest/)**: Framework to build the multi-modal retrieval system, handling data ingestion and management.
- **MultiModalVectorStoreIndex**: Enables combining visual and textual data for accurate, context-rich retrieval.
- **MoviePy**: For frame and audio extraction from YouTube videos.
- **SpeechRecognition**: For converting video audio into text for further processing.
- **LanceDB Vector Store**: To store both visual and textual data in a multimodal format, enabling efficient retrieval and interaction.
- **Streamlit**: For a beautiful and simple UI that enables users to interact with the model effortlessly.
- **YouTube API (Pytube)**: For fetching and downloading YouTube videos directly into the application.

### 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multimodal-rag-with-videos.git
   cd multimodal-rag-with-videos
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key by setting it in the environment or inputting it directly in the app.

### ⚙️ Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Enter a YouTube video link and your OpenAI API key to start the video processing.
3. Ask questions about the video content and get detailed, AI-powered responses, along with frame-by-frame analysis.

### 💡 Use Cases
- **Educational Videos**: Extract and ask questions about lectures, tutorials, or documentaries.
- **Content Review**: Analyze complex media content and get quick insights.
- **Training and Instructional Videos**: Query specific segments for training materials or step-by-step instructions.

### 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit PRs to enhance features, improve performance, or suggest new ideas.
