import streamlit as st
import os
from moviepy.editor import VideoFileClip
from pathlib import Path
import speech_recognition as sr
from pytubefix import YouTube
from PIL import Image
import matplotlib.pyplot as plt
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode
import openai
import json

# Streamlit app setup with a custom layout and title
st.set_page_config(page_title="YouTube Video Q&A with LLM", layout="centered", page_icon="🎬")

# Custom CSS styling to enhance UI
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        margin-top: 10px;
        font-size: 16px;
    }
    .stTextInput > div > input {
        background-color: #e9e9f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 YouTube Video Q&A with LLM")
st.subheader("🚀 Process and ask questions about YouTube videos effortlessly! Helpful for studying!")

st.write("Input a YouTube video link, process it, and ask questions about its content. Enjoy exploring! 🎉")
st.markdown("<hr style='border:2px solid gray'>", unsafe_allow_html=True)

# OpenAI API key input
st.write("**Step 1: Enter your OpenAI API Key**")
api_key = st.text_input("OpenAI API Key:", type="password", placeholder="sk-xxxxxxxxxxxxxxxx", help="Enter your OpenAI API key here.")
openai.api_key = os.getenv("OPENAI_API_KEY", default=api_key)

# Path configurations
output_video_path = "./video_data/"
output_folder = "./mixed_data/"
output_audio_path = "./mixed_data/output_audio.wav"
filepath = output_video_path + "input_vid.mp4"
Path(output_folder).mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

# Function to download video from YouTube
def download_video(url, output_path):
    yt = YouTube(url)
    metadata = {"Author": yt.author, "Title": yt.title, "Views": yt.views}
    yt.streams.get_highest_resolution().download(output_path=output_path, filename="input_vid.mp4")
    return metadata

# Function to extract frames from a video and save as images
def video_to_images(video_path, output_folder):
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(os.path.join(output_folder, "frame%04d.png"), fps=0.5)

# Function to extract audio from a video
def video_to_audio(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)

# Function to convert audio to text
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)
    with audio as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            text = "Audio not recognized"
        except sr.RequestError as e:
            text = f"Error: {e}"
    return text

# Function to plot images
def plot_images(image_paths):
    st.write("Frames used in the response:")
    images_shown = 0
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            st.image(image, caption=f"Frame {images_shown + 1}", use_column_width="auto")
            images_shown += 1
            if images_shown >= 7:
                break

# Retrieve query results
def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)
    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            retrieved_text.append(res_node.text)
    return retrieved_image, retrieved_text

# Initial state management
if "retriever_engine" not in st.session_state:
    st.session_state.retriever_engine = None
    st.session_state.metadata_vid = None

# Step 2: Input YouTube URL
st.markdown("### 🎥 Step 2: Enter the YouTube Video URL")
video_url = st.text_input("Enter the YouTube video link:", key="video_input", placeholder="https://www.youtube.com/watch?v=example")

# Process the video on button click
if video_url and st.session_state.retriever_engine is None:
    if st.button("🚀 Process Video"):
        try:
            with st.spinner("Processing video... This might take a while :( "):
                st.session_state.metadata_vid = download_video(video_url, output_video_path)
                video_to_images(filepath, output_folder)
                video_to_audio(filepath, output_audio_path)
                text_data = audio_to_text(output_audio_path)

                # Save extracted text to a file
                with open(output_folder + "output_text.txt", "w") as file:
                    file.write(text_data)
                os.remove(output_audio_path)

                # Set up vector stores for text and images
                text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
                image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")

                # Set up storage context for multi-modal index
                storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

                # Load documents from the output folder
                documents = SimpleDirectoryReader(output_folder).load_data()

                # Create the multi-modal index
                index = MultiModalVectorStoreIndex.from_documents(documents, storage_context=storage_context)
                st.session_state.retriever_engine = index.as_retriever(similarity_top_k=3, image_similarity_top_k=3)

                st.success("✅ Video processing completed! You can now ask questions.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Step 3: Ask questions
if st.session_state.retriever_engine:
    st.markdown("### 💬 Step 3: Ask a Question About the Video")
    user_query = st.text_input("Ask a question about the video:", key="question_input")
    
    if st.button("🔍 Submit Query") and user_query:
        try:
            img, txt = retrieve(retriever_engine=st.session_state.retriever_engine, query_str=user_query)
            image_documents = SimpleDirectoryReader(input_dir=output_folder, input_files=img).load_data()
            context_str = "".join(txt)

            # Display metadata and context
            st.write("### 📄 Video Metadata:")
            st.json(st.session_state.metadata_vid)

            st.write("### 📝 Extracted Text Context:")
            st.text(context_str)

            # Display the frames used in response
            plot_images(img)

            # Create the LLM prompt
            qa_tmpl_str = (
                "Given the provided information, including relevant images and retrieved context from the video, "
                "accurately and precisely answer the query without any additional prior knowledge.\n"
                "---------------------\n"
                "Context: {context_str}\n"
                "Metadata for video: {metadata_str}\n"
                "---------------------\n"
                "Query: {query_str}\n"
                "Answer: "
            )

            # Interact with LLM
            openai_mm_llm = OpenAIMultiModal(
                model="gpt-4-turbo", api_key=openai.api_key, max_new_tokens=1500
            )
            response_1 = openai_mm_llm.complete(
                prompt=qa_tmpl_str.format(
                    context_str=context_str, query_str=user_query, metadata_str=json.dumps(st.session_state.metadata_vid)
                ),
                image_documents=image_documents,
            )

            # Display the response
            st.markdown("### 🤖 LLM Response:")
            st.write(response_1.text)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Step 4: Process new video
if st.button("🔄 Process New Video"):
    # Reset session state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.write("App reset. Please enter a new video link.")
