import docx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import networkx as nx
import os
import pickle

nltk.download('stopwords')
nltk.download('punkt')  

folderPath = "allDocx"

# Initialize a list to store graphs for each document
documentsGraphs = []

# Iterate through the files in the folder
for filename in os.listdir(folderPath):
    # Check if the file is a .docx file
    if filename.endswith(".docx"):
        doc = docx.Document("allDocx/"+filename)
        category = filename.split('_')[0]
        record = {}
        stopWords = set(stopwords.words('english'))
        ps = PorterStemmer()
        preprocessedText = []

        text = " ".join([paragraph.text for paragraph in doc.paragraphs])
        tokens = word_tokenize(text.lower())

        for word in tokens:
            if word.isalpha() and word not in stopWords:
                stemmedWord = ps.stem(word)
                preprocessedText.append(stemmedWord)

        G = nx.DiGraph() #Directed graph
        for i in range(len(preprocessedText) - 1):
            word1 = preprocessedText[i]
            word2 = preprocessedText[i + 1]
            G.add_edge(word1, word2)
        record['category'] = category
        record['graph'] = G
        documentsGraphs.append(record)


def computeGraphDistance(graph1, graph2):
    """
    Compute the distance between two graphs based on common nodes and edges.
    """
    nodes1 = set(graph1.nodes())
    nodes2 = set(graph2.nodes())
    commonNodes = len(nodes1.intersection(nodes2))

    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    commonEdges = len(edges1.intersection(edges2))

    maxNodes = max(len(nodes1), len(nodes2))
    maxEdges = max(len(edges1), len(edges2))

    distance = 1 - (commonNodes + commonEdges) / (maxNodes + maxEdges)
    return distance


def knnClassification(testGraph, documentsGraphs, k):
    """
    Perform k-Nearest Neighbors classification for a given test graph.
    """
    distances = []
    # Compute distances between the test graph and all training graphs
    for record in documentsGraphs:
        trainGraph = record['graph']
        distance = computeGraphDistance(testGraph, trainGraph)
        distances.append((distance, record['category']))

    # Sort distances in ascending order
    distances.sort(key=lambda x: x[0])

    # Select k nearest neighbors
    nearestNeighbors = distances[:k]

    # Perform majority voting to determine the category
    categories = [neighbor[1] for neighbor in nearestNeighbors]
    categoryCounts = {category: categories.count(category) for category in set(categories)}
    predictedCategory = max(categoryCounts, key=categoryCounts.get)

    return predictedCategory

# Example usage:
testGraph = documentsGraphs[25]['graph']  # Assume the first document in documentsGraphs is the test document
k = 5  # Choose the number of nearest neighbors
predictedCategory = knnClassification(testGraph, documentsGraphs[1:], k)  # Exclude the test document from training data
print("Category Predicted By Model:", predictedCategory)
with open('trainingData.pkl', 'wb') as f:
    pickle.dump(documentsGraphs, f)
