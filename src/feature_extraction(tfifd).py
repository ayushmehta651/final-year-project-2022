from sklearn.feature_extraction.text import CountVectorizer

document = ["A MOVE to stop Mr. Gaitskell from nominating any more Labour life Peers is to be made at a meeting of Labour M Ps tomorrow.",
			"He believes that the House of Lords should be abolished and that Labour should not take any steps which would appear to an out-dated institution.",
            "A MOVE to stop Mr. Gaitskell from nominating any more Labour life Peers is to be made at a meeting of Labour M Ps tomorrow.",
            "Though they may gather some Left-wing support , a large majority of Labour M Ps are likely to turn down the Foot- Griffiths resolution.", 
            "Mr. Foot's line will be that as Labour M Ps opposed the Government Bill which brought life peers into existence.",
            "they should not now put forward nominees , He believes that the House of Lords should be abolished and that Labour should not take any steps an out- Though they may gather some Left-wing support.",
            "a large majority of Labour 0M Ps are likely to turn down the Foot-Griffiths resolution.",
            "Mr. Foot's line will be that as Labour 0M Ps opposed the Govern- ment Bill which brought life peers into existence.",
            "hey should not now put forward nominees.",
            "He believes that the House of Lords should be abolished and that Labour should not an out-dated institution.",
            "for it says that ' the Body and Blood of Christ , which are verily and indeed taken and received by the faithful in the Lord's Supper . ' It could hardly be more definite.",
            "It was at this time , too , that King James made himself immortal by becoming associated with a Bible translation - the Authorized Version ( which was never actually authorized ! ).",
            "In that year Shakespeare had just turned forty and had written Hamlet two years before . Bacon was at work and Milton was just learning to read . James was followed by Charles , in whose reign came the Scottish Prayer Book in 1637.",
            "Significantly this made a deliberate return to the Book of 1549 and became the foster mother of some of the most important Prayer Books in the Anglican Communion.",
            "Forever associated with Charles is Archbishop Laud , now so much nobler a figure than former historians led us to believe.",
            "Laud was enthusiastically hated by Calvinists and Puritans , and the sentiment was mutual.",
            "Far better to agree with a child that a particular situation is frightening , and then to face it together until the child can see how unnecessary its fears were .",
            "Because situations which may contain all the elements of fear can arise suddenly , it is a good idea to condition a child to some extent against it . To keep a child of twelve or thirteen under the impression that nothing nasty ever happens is not merely dishonest , it is unwise .",
            ]
  
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