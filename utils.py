from bs4 import BeautifulSoup
import requests
from spacy_help_functions import extract_relations, get_entities, create_entity_pairs
from collections import defaultdict

raw_text = "Bill Gates stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."

# Load spacy model
import spacy
nlp = spacy.load("en_core_web_lg")  

# Load pre-trained SpanBERT model
from spanbert import SpanBERT 
spanbert = SpanBERT("./pretrained_spanbert") 



#main function used to execute the entire process of tuple finding, passed in from main
def processQuery(service, apikey, engineID, r, threshold, q, k):
    tupleDict = defaultdict(float)
    querySet = set()
    urlSet = set()
    numIterations = 0
    hdr = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
    
    relation = ""
    entities_of_interest = [] 
    if (r == "1"):    #identify the correct relation and entities of interest based on the integer r passed in.  note that entities of interest are ordered in subject, object order.
        relation = "per:schools_attended"
        entities_of_interest = ["PERSON", "ORGANIZATION"]
    elif (r == "2"):
        relation = "per:employee_of"
        entities_of_interest = ["PERSON", "ORGANIZATION"]
    elif (r == "3"):
        relation = "per:cities_of_residence"
        entities_of_interest = ["PERSON",  "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    elif (r == "4"):
        relation = "org:top_members/employees"
        entities_of_interest = ["ORGANIZATION", "PERSON"]

    query = q
    

    print("")
    print("")
    print("")
    print("____")
    print("Parameters:")
    print("Client key	= " + apikey)
    print("Engine key	= " + engineID)
    print("Relation	= " + relation)
    print("Threshold	= " + threshold)
    print("Query		= " + query)
    print("# of Tuples	= " + k)
    print("Loading necessary liberaries; This should take a minute or so ...")

    while len(tupleDict) < int(k) and query not in querySet:   #while we are still missing tuples and we can execute a new search with a query that has not been used yet
        querySet.add(query)
        print("=========== Iteration: ", numIterations, " - Query: ", query, " ===========")
        
        numIterations += 1
        res = service.cse().list(q = query, cx = engineID,).execute()
        list_of_urls = []

        for result in res.get('items'):
            currentUrl = result.get('link')
            list_of_urls.append(currentUrl)

        urlNum = 0
        for url in list_of_urls:
            extractedAnnotations = 0
            overallRelations = 0
            extractedRelations = 0

            urlNum += 1

            print("")
            print("")
            print("URL (", urlNum, " / 10): ", url)

            if url in urlSet:      #if the url has already been visited
                print("URL already visited, skipping to next URL ...")
                continue

            urlSet.add(url)

            print("\tFetching text from url...")
            req = requests.get(url, headers = hdr)   #fetch the html contents of a webpage
            soup = BeautifulSoup(req.content, 'html.parser')  #pass content to beautifulsoup and clean up unused text
            
            for script in soup(["script", "style"]):
                script.decompose()

            text = ' '.join(soup.stripped_strings)

            if (len(text) > 20000):       #if the webpage contains more than 20000 characters, trim it
                print("\tTrimming webpage content from ", len(text), " to 20000 characters",)
                text = text[0:20000]

            print("\tWebpage length (num characters): ", len(text))
            print("\tAnnotating the webpage using spacy ...")
            doc = nlp(text)
            print("\tExtracted ", len([s for s in doc.sents]), " sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
            
            sentenceCounter = 0

            for sentence in doc.sents:    #iterate through every sentence annotated by spacy
                examples = []
                entity_pairs = create_entity_pairs(sentence, entities_of_interest)  #obtain the entity pairs using the spacy help functions

                for token, entity1, entity2 in entity_pairs:   #iterate through each entity pair obtained from spacy
                    subj = None
                    obj = None
                    if entity1[1] == entities_of_interest[0]:   #if the first entity in the entity pair tuple is the subject
                        subj = entity1
                    if entity1[1] in entities_of_interest[1:]:  #if the first entity in the entity pair tuple is the object
                        obj = entity1
                    if entity2[1] == entities_of_interest[0]:   #if the second entity in the entity pair tuple is the subject
                        subj = entity2
                    if entity2[1] in entities_of_interest[1:]:  #if the second enetity in the entity pair tuple is the object
                        obj = entity2

                    if subj != None and obj != None:            #if we identified a pair that satisfies the relation that we are analyzing, append it to our list of candidate pairs
                        examples.append({"tokens": token, "subj": subj, "obj": obj})
            
                if len(examples) > 0:                          #if there is at least 1 candidate pair that satisfies the relation that we are analyzing, extract all relations using the spacy help function
                    tupleDict, overallRelations, extractedRelations, extractedAnnotation = extract_relations(sentence, spanbert, examples, tupleDict, relation, overallRelations, extractedRelations, entities_of_interest, float(threshold))
                
                    if extractedAnnotation == True:           #if we ended up getting at least 1 annotation from the sentence extracted
                        extractedAnnotations += 1

                sentenceCounter += 1
                if (sentenceCounter % 5 == 0):
                    print("\tProcessed ", sentenceCounter, "/", len([s for s in doc.sents]), " sentences")
            
            print("\tExtracted annotations for ", extractedAnnotations, " out of total ", len([s for s in doc.sents]), " sentences")
            print("\tRelations extracted from this website: ", extractedRelations, " (Overall: ", overallRelations, ")")
            
        
        for key, value in sorted(tupleDict.items(), key= lambda x: x[1]):   #sort the tuples that currently satisfy the relation being analyzed and have at least the inputted confidence score in descending confidence score order
            currentQuery = key[0] + " " + key[2]                            #form a new query by appending the subject and object attribute values of the highest tuple that has not been used yet
            if currentQuery not in querySet:
                query = currentQuery
                break
    
        
        sortedTuple = sorted(tupleDict.items(), key= lambda x: x[1], reverse = True)   #sort the tuple dict in descending confidence score order
        print("================== ALL RELATIONS for ", relation, " ( ", len(sortedTuple), " ) =================")
        
        for t in sortedTuple:
            print("Confidence: ", t[1], " 		| Subject: ", t[0][0], " 		| Object: ", t[0][2])
        
        print("Number of Iterations: ", numIterations)

        if query in querySet and len(sortedTuple) < k:                            #if the current query has already been used and we have not yet acquired the correct amount of tuples
            print("The execution stalled without retrieving the desired number of tuples with the specified confidence.")
        

        
