import tkinter as tk
from tkinter import filedialog, messagebox
import google.generativeai as genai
from PIL import Image
import pytesseract
from vosk import Model, KaldiRecognizer
import pyaudio
import json

# Path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Initialize the Vosk model (change this to the path of your downloaded model)
model_path = r'C:\\vosk-model\\vosk-model-en-us-0.22'
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

# Initialize the generative AI model
ai_model = genai.GenerativeModel("gemini-1.5-flash")

# Functions
def extract_text_from_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", ".png;.jpg;.jpeg;.bmp;*.tiff")]
    )
    if not file_path:
        return
    try:
        image = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(image).strip()
        if extracted_text:
            entry.delete("1.0", tk.END)
            entry.insert(tk.END, extracted_text)
            messagebox.showinfo("Success", "Text extracted successfully!")
        else:
            messagebox.showwarning("No Text Found", "No text found in the image.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def check_news():
    news_content = entry.get("1.0", tk.END).strip()
    if not news_content:
        messagebox.showwarning("Input Error", "Please enter some news content.")
        return
    try:
        response = ai_model.generate_content(
            f"Is this news Real or Fake? Give one word answer and explain. News: {news_content}"
        )
        result_label.config(text=f"Result: {response.text.strip()}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def listen_speech_to_text():
    # Listen to audio using Vosk
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=4000)
    stream.start_stream()
   
    messagebox.showinfo("Listening", "Listening for your speech... Please speak now.")


    while True:
        data = stream.read(4000)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            result_json = json.loads(result)
            recognized_text = result_json['text']
            entry.delete("1.0", tk.END)
            entry.insert(tk.END, recognized_text)
            messagebox.showinfo("Speech Recognized", f"Text: {recognized_text}")
            break
        else:
            partial_result = recognizer.PartialResult()
            partial_result_json = json.loads(partial_result)
            print("Partial Result:", partial_result_json['partial'])


# App Window
root = tk.Tk()
root.title("Fake News Detector")
root.geometry("900x650")
root.configure(bg="#F9F7F3")

# Custom Button Style
def create_button(master, text, command, bg, fg, hover_bg):
    def on_enter(e):
        button["bg"] = hover_bg

    def on_leave(e):
        button["bg"] = bg

    button = tk.Button(
        master, text=text, command=command, font=("Poppins", 14, "bold"),
        bg=bg, fg=fg, bd=0, padx=20, pady=10, relief="flat"
    )
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    return button


# Title
tk.Label(
    root, text="📰 Fake News Detector 📰", font=("Poppins", 32, "bold"),
    fg="#333", bg="#F9F7F3"
).pack(pady=30)


# Text Entry Box
entry_frame = tk.Frame(root, bg="#FFF", bd=5, relief="solid")
entry_frame.pack(pady=20)

entry = tk.Text(
    entry_frame, height=8, width=60, font=("Poppins", 14),
    bg="#E9FBE9", fg="#333", wrap="word", bd=0, padx=10, pady=10
)
entry.pack(fill=tk.BOTH)


# Button Frame
button_frame = tk.Frame(root, bg="#F9F7F3")
button_frame.pack(pady=20)

upload_button = create_button(
    button_frame, "📷 Upload Image", extract_text_from_image, "#F9D1C2", "#333", "#F8B8A3"
)
upload_button.grid(row=0, column=0, padx=30)

check_button = create_button(
    button_frame, "✅ Check News", check_news, "#B0DDF7", "#333", "#8AC3F1"
)
check_button.grid(row=0, column=1, padx=30)

speech_button = create_button(
    button_frame, "🎤 Audio", listen_speech_to_text, "#FFEB3B", "#333", "#FBC02D"
)
speech_button.grid(row=0, column=2, padx=30)


# Result Label
result_label = tk.Label(
    root, text="Result: ", font=("Poppins", 16, "bold"),
    fg="#333", bg="#F9F7F3", wraplength=750, justify="center"
)
result_label.pack(pady=30)

# Run the App
root.mainloop()