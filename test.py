import docx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import networkx as nx
import os
import pickle

nltk.download('stopwords')
nltk.download('punkt')

def preprocessText(text):
    stopWords = set(stopwords.words('english'))
    ps = PorterStemmer()
    preprocessedText = []

    tokens = word_tokenize(text.lower())

    for word in tokens:
        if word.isalpha() and word not in stopWords:
            stemmedWord = ps.stem(word)
            preprocessedText.append(stemmedWord)

    return preprocessedText

def createGraphRepresentation(text):
    G = nx.DiGraph()

    for i in range(len(text) - 1):
        word1 = text[i]
        word2 = text[i + 1]
        G.add_edge(word1, word2)

    return G

with open('trainingData.pkl', 'rb') as f:
    trainingData = pickle.load(f)

def computeGraphDistance(graph1, graph2):
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

def classifyWithKnn(testGraph, trainingData, k):
    distances = []
    for record in trainingData:
        trainGraph = record['graph']
        distance = computeGraphDistance(testGraph, trainGraph)
        distances.append((distance, record['category']))

    distances.sort(key=lambda x: x[0])

    nearestNeighbors = distances[:k]

    categories = [neighbor[1] for neighbor in nearestNeighbors]
    categoryCounts = {category: categories.count(category) for category in set(categories)}
    predictedCategory = max(categoryCounts, key=categoryCounts.get)
    return predictedCategory

testFolderPath = "testDocx"

totalTestDocuments = 0
correctPredictions = 0

for filename in os.listdir(testFolderPath):
    if filename.endswith(".docx"):
        testDoc = docx.Document(os.path.join(testFolderPath, filename))
        trueCategory = filename.split('_')[0]
        print("Actual:"+trueCategory)
        testText = " ".join([paragraph.text for paragraph in testDoc.paragraphs])
        preprocessedTestText = preprocessText(testText)
        testGraph = createGraphRepresentation(preprocessedTestText)
        k = 5
        predict = classifyWithKnn(testGraph, trainingData, k)
        predictedCategory = predict.split('.')[0]
        print("Predicted:"+predictedCategory)
        totalTestDocuments += 1
        if predictedCategory == trueCategory:
            correctPredictions += 1

accuracy = ((correctPredictions) / totalTestDocuments) * 100 

print("Accuracy:", accuracy)
