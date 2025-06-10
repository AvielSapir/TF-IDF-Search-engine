import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import os
import string
import math

FOLDER_DIR = "documents"


def loadDocuments(DIR):
    documents = []
    docNames = []
    for filename in os.listdir(DIR):
        if filename.endswith(".txt"):
            docNames.append(filename)
            filePath = os.path.join(DIR, filename)
            with open(filePath, "r", encoding="utf-8") as file:
                documents.append(file.read())
    return (documents, docNames)

def processText(text):
    text = text.lower()
    tokens = word_tokenize(text)

    cleaned_words = []
    for token in tokens:
        token = token.translate(str.maketrans('', '', string.punctuation))
        if token.isalpha() and token != "":
            cleaned_words.append(token)

    filtered_words = []
    for token in cleaned_words:
        if token not in set(stopwords.words('english')):
            filtered_words.append(token)


    return filtered_words

def buildVocabulary(documents):
    vocabulary = {}
    wordI = 0
    for document in documents:
        for word in document:
            if word not in vocabulary.keys():
                vocabulary[word] = wordI
                wordI += 1
    return vocabulary


def calculateTF(doc, vocabulary):
    tfVec = {}
    numOfWords = len(doc)

    if numOfWords == 0:
        return {}

    counter = {}
    for word in doc:
        counter[word] = counter.get(word, 0) + 1

    for word, count in counter.items():
        if word in vocabulary.keys():
            word_index = vocabulary[word]
            tfVec[word_index] = count / numOfWords

    return tfVec

def calculateIDF(document, vocabulary):
    idfValues = {}
    numOfDoc = len(document)

    counter = {}
    for word in vocabulary.values():
        counter[word] = 0

    for doc in document:
        for word in set(doc):
            if word in vocabulary:
                counter[vocabulary[word]] += 1

    for word, count in counter.items():
        if count > 0:
            idfValues[word] = math.log(numOfDoc / count)
        else:
            idfValues[word] = 0
    return idfValues

def calculateTfIdfMatrix(tfVector, idfValues, vocSize):
    tfidfMatrix = []
    for vect in tfVector:
        docVector = [0.0] * vocSize
        for wordI, tf in vect.items():
            idf = idfValues.get(wordI, 0)
            tfidf = tf * idf
            docVector[wordI] = tfidf
        tfidfMatrix.append(docVector)
    return tfidfMatrix

def TFIDF(documents, vocabulary):
    tfVector = []
    for document in documents:
        tfVector.append(calculateTF(document, vocabulary))
    idfValues = calculateIDF(documents, vocabulary)
    print("[*] finished calculating TF-IDF scores")
    return idfValues ,calculateTfIdfMatrix(tfVector, idfValues, len(vocabulary))

def cosineSimilarity(vec1, vec2):
    dot_product = 0.0
    norm_vec1 = 0.0
    norm_vec2 = 0.0

    for i in range(len(vec1)):
        dot_product += vec1[i] * vec2[i]
        norm_vec1 += vec1[i] ** 2
        norm_vec2 += vec2[i] ** 2

    norm_vec1 = math.sqrt(norm_vec1)
    norm_vec2 = math.sqrt(norm_vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)

# def searchFor():


# main
print("[*] Loading documents...")
documents, filenames = loadDocuments(FOLDER_DIR)

if not documents:
    print(f"[!] Not found!")

processedDocuments = []
for doc in documents:
    processedDocuments.append(processText(doc))


sum = 0
for doc in processedDocuments:
    sum += len(doc)
print(sum, "[#] words found ")

vocabulary = buildVocabulary(processedDocuments)
print(len(vocabulary), "[#] unique words found ")

idfValues, tfidfMatrix = TFIDF(processedDocuments, vocabulary)


while True:
    query = input("\n[>] search for: ")
    if query.lower() == "exit":
        break
    if not query:
        print("[!] not searchable! ")
        continue

    print("[*] searching...")
    processedQuery = processText(query)
    if not processedQuery:
        print("[!] Try more descriptive words! ")
    queryTfVector = calculateTF(processedQuery, vocabulary)
    queryTfIdfVector = [0.0] * len(vocabulary)
    for wordI, tf_value in queryTfVector.items():
        idf_value = idfValues.get(wordI, 0.0)
        queryTfIdfVector[wordI] = tf_value * idf_value

    searchResults = []
    for i, docTfidfVector in enumerate(tfidfMatrix):
        similarityScore = cosineSimilarity(queryTfIdfVector, docTfidfVector)
        if similarityScore > 0:
            searchResults.append((similarityScore, filenames[i]))
    searchResults.sort(key=lambda x: x[0], reverse=True)


    print("\n[*] Search Results:")
    if searchResults:
        sum = 0
        for score, filename in searchResults[:5]:
            sum += score



        for score, filename in searchResults[:5]:
            print(f"     | {filename}: Similarity = {(100*score) / sum:.2f}%")
        if len(searchResults) > 5:
            print(f"  ({len(searchResults) - 5} more results...)")
    else:
        print("[!] No relevant documents found for your query.")