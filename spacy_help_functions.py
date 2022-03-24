import spacy
from collections import defaultdict

spacy2bert = { 
        "ORG": "ORGANIZATION",
        "PERSON": "PERSON",
        "GPE": "LOCATION", 
        "LOC": "LOCATION",
        "DATE": "DATE"
        }

bert2spacy = {
        "ORGANIZATION": "ORG",
        "PERSON": "PERSON",
        "LOCATION": "LOC",
        "CITY": "GPE",
        "COUNTRY": "GPE",
        "STATE_OR_PROVINCE": "GPE",
        "DATE": "DATE"
        }


def get_entities(sentence, entities_of_interest):
    return [(e.text, spacy2bert[e.label_]) for e in sentence.ents if e.label_ in spacy2bert]


def extract_relations(doc, spanbert, examples, res, relation_of_interest, overallRelations, extractedRelations, entities_of_interest=None, conf=0.7):
    #num_sentences = len([s for s in doc.sents])
    #print("Total # sentences = {}".format(num_sentences))
    #res = defaultdict(int)
    #for sentence in doc.sents:
        #print("\tprocessing sentence: {}".format(sentence))
        #entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        #examples = []
        #for ep in entity_pairs:
            #examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            #examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
    
    extractedAnnotation = False   #boolean value used to track if we extracted any annotations from the sentence being analyzed (used only for )
    preds = spanbert.predict(examples)   #run the spanbert model on every candidate pair passed in from the parameter input
    for ex, pred in list(zip(examples, preds)):
        relation = pred[0]
        if relation != relation_of_interest:   #if we encounter a relation that is not of interest to us, skip it
            continue
        extractedAnnotation = True
        overallRelations += 1
        subj = ex["subj"][0]
        obj = ex["obj"][0]
        confidence = pred[1]
        if confidence > conf:                #if the current relation is at least greater than the confidence passed in by the parameter
            if (subj, relation, obj) not in res:   #first check to see if the relation is already in our list of tuples, if not then append it to the res dictionary
                res[(subj, relation, obj)] = confidence
                extractedRelations += 1
                print("\n\t\t=== Extracted Relation ===")
                print("\t\tInput tokens: {}".format(ex['tokens']))
                print("\t\tRelation: {} (Output Confidence: {:.3f}) ; Subject: {} ; Object: {}".format(relation, confidence, subj, obj))
                print("\t\tAdding to set of extracted relations")
                print("\t\t==========")
            elif res[(subj, relation, obj)] < confidence:    #if the current relation tuple has already been recorded, but this version contains a higher confidence leve, then update it
                overallRelations -= 1
                res[(subj, relation, obj)] = confidence
            else:
                print("\n\t\t=== Extracted Relation ===")   #otherwise ignore the duplicate with lower confidence value
                print("\t\tInput tokens: {}".format(ex['tokens']))
                print("\t\tRelation: {} (Output Confidence: {:.3f}) ; Subject: {} ; Object: {}".format(relation, confidence, subj, obj))
                print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
                print("\t\t==========")
        else: 
            print("\n\t\t=== Extracted Relation ===")     #if we've extracted a relation with a lower confidence value than the confidence value passed in by our parameter, ignore it
            print("\t\tInput tokens: {}".format(ex['tokens']))
            print("\t\tRelation: {} (Output Confidence: {:.3f}) ; Subject: {} ; Object: {}".format(relation, confidence, subj, obj))
            print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
            print("\t\t==========")
            
        
    return res, overallRelations, extractedRelations, extractedAnnotation   #return the updated dictionary of tuples, the overall relations taken from the sentence, extracted relations from the sentence, and whether the sentence had been annotated or not


def create_entity_pairs(sents_doc, entities_of_interest, window_size=40):
    '''
    Input: a spaCy Sentence object and a list of entities of interest
    Output: list of extracted entity pairs: (text, entity1, entity2)
    '''
    if entities_of_interest is not None:
        entities_of_interest = {bert2spacy[b] for b in entities_of_interest}
    ents = sents_doc.ents # get entities for given sentence

    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if entities_of_interest is not None and e1.label_ not in entities_of_interest:
            continue

        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if entities_of_interest is not None and e2.label_ not in entities_of_interest:
                continue
            if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
                continue

            if (1 <= (e2.start - e1.end) <= window_size):

                punc_token = False
                start = e1.start - 1 - sents_doc.start
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0

                # Find end of sentence
                punc_token = False
                start = e2.end - sents_doc.start
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc

                if (right_r - left_r) > window_size: # sentence should not be longer than window_size
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (e1.text, spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (e2.text, spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if e1.start == e1.end:
                    assert x[e1.start-gap] == e1.text, "{}, {}".format(e1_info, x)
                if e2.start == e2.end:
                    assert x[e2.start-gap] == e2.text, "{}, {}".format(e2_info, x)
                entity_pairs.append((x, e1_info, e2_info))
    return entity_pairs

