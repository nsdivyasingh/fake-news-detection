import tkinter as tk
from tkinter import filedialog, messagebox
import google.generativeai as genai
from PIL import Image
import pytesseract
from vosk import Model, KaldiRecognizer
import pyaudio
import json
import csv
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Configure API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
ai_model = genai.GenerativeModel("gemini-1.5-flash")

# Path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tess_path):
    pytesseract.pytesseract.tesseract_cmd = tess_path
else:
    raise FileNotFoundError("Tesseract not found at the specified path.")

# Initialize Vosk model
def initialize_vosk():
    global recognizer, vosk_model
    vosk_model = Model(r'C:\\vosk-model\\vosk-model-en-us-0.22')
    recognizer = KaldiRecognizer(vosk_model, 16000)

# CSV Setup
file_name = "metrics.csv"
fieldnames = ["News", "Actual Label", "Predicted Label"]

if not os.path.exists(file_name):
    with open(file_name, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

def save_article(news_content, actual_label, predicted_label):
    with open(file_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({"News": news_content, "Actual Label": actual_label, "Predicted Label": predicted_label})

# Metrics lists
true_labels = []
predicted_labels = []

def extract_text_from_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;.jpeg;.bmp;*.tiff")])
    if file_path:
        try:
            extracted_text = pytesseract.image_to_string(Image.open(file_path)).strip().lower()
            entry.delete("1.0", tk.END)
            entry.insert(tk.END, extracted_text)
            messagebox.showinfo("Success", "Text extracted successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def check_news():
    news_content = entry.get("1.0", tk.END).strip()
    if not news_content:
        messagebox.showwarning("Input Error", "Please enter some news content.")
        return
    try:
        response = ai_model.generate_content(
            f"Is this news Real or Fake? Give a one-word answer (Real/Fake) and a short explanation. News: {news_content}"
        )
        result_text = response.text.strip().lower()

        if "real" in result_text and "fake" in result_text:
            predicted_label = "real" if result_text.index("real") < result_text.index("fake") else "fake"
        elif "real" in result_text:
            predicted_label = "real"
        elif "fake" in result_text:
            predicted_label = "fake"
        else:
            messagebox.showerror("Error", "AI did not return a valid label (Real/Fake).")
            return

        actual_label = actual_label_var.get().lower()
        true_labels.append(actual_label)
        predicted_labels.append(predicted_label)
        result_label.config(text=f"Result: {predicted_label.capitalize()}\n\n{response.text.strip()}")
        save_article(news_content, actual_label, predicted_label)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def listen_speech_to_text():
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        stream.start_stream()
        messagebox.showinfo("Listening", "Speak now.")
        captured_text = ""
        for _ in range(50):
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result_json = json.loads(recognizer.Result())
                captured_text = result_json.get('text', '')
                break
        stream.stop_stream()
        stream.close()
        p.terminate()
        if captured_text:
            entry.delete("1.0", tk.END)
            if entry.get("1.0", tk.END).strip():
                if not messagebox.askyesno("Overwrite", "Replace existing text?"):
                    return
            entry.insert(tk.END, captured_text)
            messagebox.showinfo("Speech Recognized", f"Text: {captured_text}")
        else:
            messagebox.showwarning("No Speech Detected", "No text detected from speech.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_metrics():
    if len(true_labels) < 2:
        messagebox.showwarning("Not Enough Data", "Need at least 2 samples to compute metrics.")
        return

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label="real")
    recall = recall_score(true_labels, predicted_labels, pos_label="real")
    f1 = f1_score(true_labels, predicted_labels, pos_label="real")

    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}
    metrics_text = "\n".join([f"{k}: {v:.2%}" for k, v in metrics.items()])
    messagebox.showinfo("Model Performance", metrics_text)

    # Plot metrics
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Fake News Detector Model Metrics")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=["real", "fake"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Classification Report
    report = classification_report(true_labels, predicted_labels, target_names=["real", "fake"])
    with open("classification_report.txt", "w") as f:
        f.write(report)
    messagebox.showinfo("Report Saved", "Classification report saved to classification_report.txt.")

def upload_csv_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                actual = row["Actual Label"].strip().lower()
                predicted = row["Predicted Label"].strip().lower()
                true_labels.append(actual)
                predicted_labels.append(predicted)
        messagebox.showinfo("Success", "CSV data uploaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Tooltip helper
def add_tooltip(widget, text):
    def on_enter(e):
        tip.place(x=e.x_root - root.winfo_rootx() + 10, y=e.y_root - root.winfo_rooty() + 10)
        tip.config(text=text)
    def on_leave(e):
        tip.place_forget()
    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)

# App UI
root = tk.Tk()
root.title("Fake News Detector")
root.geometry("950x750")
root.configure(bg="#F9F7F3")

# Tooltip label
tip = tk.Label(root, text="", bg="#FFF8DC", font=("Poppins", 10), relief="solid", bd=1)

# Button style
def create_button(master, text, command, bg, fg, hover_bg):
    def on_enter(e): button["bg"] = hover_bg
    def on_leave(e): button["bg"] = bg
    button = tk.Button(
        master, text=text, command=command, font=("Poppins", 14, "bold"),
        bg=bg, fg=fg, bd=0, padx=20, pady=10, relief="flat"
    )
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    return button

# Title
tk.Label(root, text="📰 Fake News Detector 📰", font=("Poppins", 32, "bold"), fg="#333", bg="#F9F7F3").pack(pady=30)

# Text Entry
entry_frame = tk.Frame(root, bg="#FFF", bd=5, relief="solid")
entry_frame.pack(pady=10)

entry = tk.Text(
    entry_frame, height=8, width=60, font=("Poppins", 14),
    bg="#E9FBE9", fg="#333", wrap="word", bd=0, padx=10, pady=10
)
entry.pack(fill=tk.BOTH)

# Button Frame
button_frame = tk.Frame(root, bg="#F9F7F3")
button_frame.pack(pady=20)

upload_button = create_button(button_frame, "📷 Upload Image", extract_text_from_image, "#F9D1C2", "#333", "#F8B8A3")
upload_button.grid(row=0, column=0, padx=10)
add_tooltip(upload_button, "Upload an image with news text")

check_button = create_button(button_frame, "✅ Check News", check_news, "#B0DDF7", "#333", "#8AC3F1")
check_button.grid(row=0, column=1, padx=10)
add_tooltip(check_button, "Use AI to classify the news")

speech_button = create_button(button_frame, "🎤 Audio", listen_speech_to_text, "#FFEB3B", "#333", "#FBC02D")
speech_button.grid(row=0, column=2, padx=10)
add_tooltip(speech_button, "Use your voice to input news")

metrics_button = create_button(button_frame, "📊 Show Metrics", show_metrics, "#C1E1C1", "#333", "#A7D7A7")
metrics_button.grid(row=0, column=3, padx=10)
add_tooltip(metrics_button, "View and evaluate performance metrics")

# # Actual Label Selector
# label_frame = tk.Frame(root, bg="#F9F7F3")
# label_frame.pack(pady=10)

# actual_label_var = tk.StringVar(value="real")
# tk.Label(label_frame, text="Actual Label:", font=("Poppins", 14), bg="#F9F7F3").pack(side="left", padx=10)
# tk.Radiobutton(label_frame, text="Real", variable=actual_label_var, value="real", font=("Poppins", 12), bg="#F9F7F3").pack(side="left")
# tk.Radiobutton(label_frame, text="Fake", variable=actual_label_var, value="fake", font=("Poppins", 12), bg="#F9F7F3").pack(side="left")

# Result Label
result_label = tk.Label(root, text="Result: ", font=("Poppins", 16, "bold"), fg="#333", bg="#F9F7F3", wraplength=750, justify="center")
result_label.pack(pady=20)

# Initialize Vosk after UI loads
root.after(1000, initialize_vosk)

# Run App
root.mainloop()