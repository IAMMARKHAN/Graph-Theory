import docx
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import networkx as nx
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text and create graph representation
def preprocess_and_create_graph(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    preprocessed_text = [word for word in tokens if word.isalpha() and word not in stop_words]

    G = nx.DiGraph()  # Directed graph
    for i in range(len(preprocessed_text) - 1):
        word1 = preprocessed_text[i]
        word2 = preprocessed_text[i + 1]
        G.add_edge(word1, word2)
    return G

# Function to visualize directed graph
def visualize_graph(graph):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='k', linewidths=1, arrowsize=20)
    plt.title("Directed Graph Visualization")
    plt.show()

# Function to handle file selection and processing
def select_and_process_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(filetypes=[("Word files", "*.docx")])
    if file_path:
        process_docx_file(file_path)
    else:
        print("No file selected.")

# Browse a .docx file, preprocess text, and visualize graph
def process_docx_file(file_path):
    doc = docx.Document(file_path)
    text = " ".join([paragraph.text for paragraph in doc.paragraphs])
    graph = preprocess_and_create_graph(text)
    visualize_graph(graph)

# Example usage
if __name__ == "__main__":
    select_and_process_file()
