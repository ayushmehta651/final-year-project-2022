from sklearn.feature_extraction.text import CountVectorizer
  
document = ["A move to stop Mr. Gaitskell from",
            "Foods would still be free in families receving",
            "That they cannot pay more than 357 million"]
  
# Create a Vectorizer Object
vectorizer = CountVectorizer()
  
vectorizer.fit(document)
  
# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)
  
# Encode the Document
vector = vectorizer.transform(document)
  
# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector.toarray())