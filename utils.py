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




def processQuery(service, apikey, engineID, r, threshold, q, k):
    tupleDict = defaultdict(float)
    querySet = set()
    urlSet = set()
    numIterations = 0
    hdr = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
    
    relation = ""
    entities_of_interest = []
    if (r == "1"):
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

    while len(tupleDict) < int(k) and query not in querySet:
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

            if url in urlSet:
                print("URL already visited, skipping to next URL ...")
                continue

            urlSet.add(url)

            print("\tFetching text from url...")
            req = requests.get(url, headers = hdr)
            soup = BeautifulSoup(req.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()

            text = ' '.join(soup.stripped_strings)

            if (len(text) > 20000):
                print("\tTrimming webpage content from ", len(text), " to 20000 characters",)
                text = text[0:20000]

            print("\tWebpage length (num characters): ", len(text))
            print("\tAnnotating the webpage using spacy ...")
            doc = nlp(text)
            print("\tExtracted ", len([s for s in doc.sents]), " sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
            
            sentenceCounter = 0

            for sentence in doc.sents:
                examples = []
                entity_pairs = create_entity_pairs(sentence, entities_of_interest)

                for token, entity1, entity2 in entity_pairs:
                    subj = None
                    obj = None
                    if entity1[1] == entities_of_interest[0]:
                        subj = entity1
                    if entity1[1] in entities_of_interest[1:]:
                        obj = entity1
                    if entity2[1] == entities_of_interest[0]:
                        subj = entity2
                    if entity2[1] in entities_of_interest[1:]:
                        obj = entity2

                    if subj != None and obj != None:
                        examples.append({"tokens": token, "subj": subj, "obj": obj})
            
                if len(examples) > 0:
                    tupleDict, overallRelations, extractedRelations, extractedAnnotation = extract_relations(sentence, spanbert, examples, tupleDict, relation, overallRelations, extractedRelations, entities_of_interest, float(threshold))
                
                    if extractedAnnotation == True:
                        extractedAnnotations += 1

                sentenceCounter += 1
                if (sentenceCounter % 5 == 0):
                    print("\tProcessed ", sentenceCounter, "/", len([s for s in doc.sents]), " sentences")
            
            print("\tExtracted annotations for ", extractedAnnotations, " out of total ", len([s for s in doc.sents]), " sentences")
            print("\tRelations extracted from this website: ", extractedRelations, " (Overall: ", overallRelations, ")")
            
        
        for key, value in sorted(tupleDict.items(), key= lambda x: x[1]):
            currentQuery = key[0] + " " + key[2]
            if currentQuery not in querySet:
                query = currentQuery
                break
    
        
        sortedTuple = sorted(tupleDict.items(), key= lambda x: x[1], reverse = True)
        print("================== ALL RELATIONS for ", relation, " ( ", len(sortedTuple), " ) =================")
        
        for t in sortedTuple:
            print("Confidence: ", t[1], " 		| Subject: ", t[0][0], " 		| Object: ", t[0][2])
        
        print("Number of Iterations: ", numIterations)

        if query in querySet and len(sortedTuple) < k:
            print("The execution stalled without retrieving the desired number of tuples with the specified confidence.")
        

        
