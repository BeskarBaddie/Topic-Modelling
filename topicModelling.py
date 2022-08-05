
import shutup






from contextlib import nullcontext
from datetime import datetime
import math
from operator import itemgetter
import os
import random
import re
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


from collections.abc import Mapping
from gensim import topic_coherence

import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
os.getcwd

import pickle
import nltk
import spacy

#from collections import Mapping


lda_model=nullcontext

#from gensim.models import ldamodel

import gensim
import gensim.corpora as corpora


from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel 
from gensim.utils import wraps 



from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter, OrderedDict

from pprint import pprint

import numpy as np




## Folder setup:
BasePath   ='/Users/Administrator/Desktop/LDA Topics Modeling/'


MalletPath      = BasePath + 'Mallet/'
MalletStopPath  = MalletPath + 'stops/'
MalletBinPath   = MalletPath + 'mallet-2.0.8/bin/mallet'
MalletStopWords = set([word.strip() for word in open(MalletStopPath + 'APPAREL.txt', 'r',encoding="utf-16").readlines()])
#MalletStopWords = set([word.strip() for word in open(MalletStopPath + 'en.txt', 'r',encoding="utf8").readlines()])

########################
DatasetsPath = BasePath  + 'Industry Files - Raw Data/'
StopsPath    = BasePath  + '/stops/' 
ResultsPath  = BasePath  + 'results/'
#############


#ProjectName  = 'csr'
#DatasetsPath = BasePath + ProjectName + '/AEROSPACE DEFENSE/'
#StopsPath    = BasePath + ProjectName + '/stops/' 
#ResultsPath  = BasePath + ProjectName + '/results/'

#ReportFolderNames = {'AR':'Annual Reports', 'CSR':'CSR Reports'}
ReportFolderNames = {'AR':'Annual Reports', 'CSR':'CSR Reports','Reports':'Reports'}



## Function:
##    - Get the folder names of the entire industries.
def getDataFolders():
    
    return os.listdir(DatasetsPath)


## Function:
##   - Get the filenames of AR and CSR reports of one dataset.
def getDataFilenames(dataset):
    #dataFilenames = {'AR':{'all':set()}, 'CSR':{'all':set()}}

    dataFilenames = {'Reports':{'all':set()}}
    
    # Read all reports.


    for report in ['Reports']:  #CHANGE BACK TO AR CSR 
        for folderPath, folders, files in os.walk(DatasetsPath + dataset + '/' + ReportFolderNames[report]): 
            #CHANGE THIS BACK FOR PART 2                 
        #for folderPath, folders, files in os.walk(DatasetsPath + dataset + '/' + report):
            # Skip the folder that does not consist of any txt data files.
            if not files:
                continue

            # Parse the yearly datafiles.
            try:
                # When there exists specific year folder,
                year = int(folderPath[folderPath.rfind('/') + 1:])
                filenames = dataFilenames[report].setdefa0ult(year, set())        
                for file in sorted(files):
                    if not file.endswith('.txt'):
                        continue
                    fullFilename = folderPath + '/' + file

                    # Check whether it contains some dummies.
                    dummyLeftIndex = fullFilename.rfind('_')
                    dummyRightIndex = dummyLeftIndex + fullFilename[dummyLeftIndex:].find('.')
                    if fullFilename[dummyLeftIndex + 1:dummyRightIndex].isnumeric():
                        if (fullFilename[:dummyLeftIndex] + fullFilename[dummyRightIndex:]) in filenames:
                            continue

                    # Add to the corpus files.                    
                    filenames.add(fullFilename)
                    dataFilenames[report]['all'].add(fullFilename)
            except:
                # When there is no specific year folder,
                for file in sorted(files):
                    if not file.endswith('.txt'):
                        continue                    
                    year = int(re.search("(\d{4})", file).group(1))
                    filenames = dataFilenames[report].setdefault(year, set())        
                    fullFilename = folderPath + '/' + file

                    # Check whether it contains some dummies like 001.
                    dummyLeftIndex = fullFilename.rfind('_')
                    dummyRightIndex = dummyLeftIndex + fullFilename[dummyLeftIndex:].find('.')
                    if fullFilename[dummyLeftIndex + 1:dummyRightIndex].isnumeric():
                        if (fullFilename[:dummyLeftIndex] + fullFilename[dummyRightIndex:]) in filenames:
                            continue

                    # Add to the corpus files.                                        
                    filenames.add(fullFilename)
                    dataFilenames[report]['all'].add(fullFilename)
    
    return dataFilenames


## Function:
##   - Count word frequencies for a collection of text documents.
def countNgrams(filenames, pickle_filename):
    unigramTermFreqs, bigramTermFreqs, trigramTermFreqs = Counter(), Counter(), Counter()
    unigramDocFreqs, bigramDocFreqs, trigramDocFreqs = Counter(), Counter(), Counter()
    
    # For each document with the given filename,
    numDocuments = len(filenames)
    print("  + Counting ngrams...")
    for (n, filename) in enumerate(filenames):
        # Process unigram and bigrams.
        print("    - Working on %d/%d documents!" % (n+1, numDocuments), end='\r')
        with open(filename, 'r', encoding='ISO-8859-1') as file:
            # Read the document as one/two sequence of word tokens.
            tokens = nltk.word_tokenize(file.read())
            unigramCounts = Counter(tokens)
            bigramCounts = Counter(ngrams(tokens, 2))
            trigramCounts = Counter(ngrams(tokens, 3))

            # Accumulate the actual counts of consecutive words.
            unigramTermFreqs.update(unigramCounts)
            bigramTermFreqs.update(bigramCounts)
            trigramTermFreqs.update(trigramCounts)

            # Accumulate the exisentece of consecutive words.
            unigramDocFreqs.update(Counter(unigramCounts.keys()))
            bigramDocFreqs.update(Counter(bigramCounts.keys()))
            trigramDocFreqs.update(Counter(trigramCounts.keys()))

    # Store the ngrams into files if necessary.
    #if pickle_filename: commented out
        pickle.dump(unigramTermFreqs, open(ResultsPath + '%s_unigram-tfs.Counter' % pickle_filename, 'wb'))
        pickle.dump(unigramDocFreqs, open(ResultsPath + '%s_unigram-dfs.Counter' % pickle_filename, 'wb'))
        pickle.dump(bigramTermFreqs, open(ResultsPath + '%s_bigram-tfs.Counter' % pickle_filename, 'wb'))
        pickle.dump(bigramDocFreqs, open(ResultsPath + '%s_bigram-dfs.Counter' % pickle_filename, 'wb'))
        pickle.dump(trigramTermFreqs, open(ResultsPath + '%s_trigram-tfs.Counter' % pickle_filename, 'wb'))
        pickle.dump(trigramDocFreqs, open(ResultsPath + '%s_trigram-dfs.Counter' % pickle_filename, 'wb'))

    return (unigramTermFreqs, unigramDocFreqs), (bigramTermFreqs, bigramDocFreqs), (trigramTermFreqs, trigramDocFreqs)
 

## Function:
##   - Execute countNgrams for all industries in the folder.
def countNgrams_batch(datasets=getDataFolders()):    
    # For each dataset,
    for dataset in datasets:
        # Count term-frequencies and document-frequencies of unigrams and bigrams.
        print('+ Reading the industry corpus for [%s]...' % dataset)
        dataFilenames = getDataFilenames(dataset)
        for report in ['AR', 'CSR']:
            countNgrams(sorted(list(dataFilenames[report]['all'])), '%s_%s' % (dataset, report))




def trigram2Dict(dataset, report):
    triTfs = pickle.load(open(ResultsPath + '%s_%s_trigram-tfs.Counter' % (dataset, report), 'rb'))   
    trigramDict = {}
    for (word0, word1, word2) in triTfs:
        bigramDict = trigramDict.setdefault((word0, word1), {})
        bigramDict[word2] = triTfs[(word0, word1, word2)]
    return trigramDict

## Function:
##   - Check whether a n-gram exhibits a consistent patterns:
##   - Only the first character is capitalized or all caps.
def checkCapitalConsistency(ngram):        
    isOnlyFirst, isAllCaps = True, True
    for word in ngram:
        # Skip the stop words.
        if word.lower() in MalletStopWords:
            return False

        # Check the capitalization of the first letter.
        isOnlyFirst = isOnlyFirst.__and__(word[0].isupper())
        for letter in word[1:]:            
            isOnlyFirst = isOnlyFirst.__and__(letter.islower())
        isAllCaps = isAllCaps.__and__(word.isupper())
    return isOnlyFirst.__or__(isAllCaps)


## Function:
##    - Merge multi-word proper names for an industry.
def mergeCapitalNgrams(dataset):
    mergedNgrams = {}

    outputFile = open(ResultsPath + '%s_capitalNgrams.txt' % dataset, 'w')
    for report in['AR', 'CSR']:
        outputFile.write('+ Regarding the [%s]...\n' % ReportFolderNames[report])

       


        biTfs = pickle.load(open(ResultsPath + '%s_%s_bigram-tfs.Counter' % (dataset, report), 'rb'))
        triDict = trigram2Dict(dataset, report)
        for bigram in sorted(biTfs, key=biTfs.get, reverse=True):
            # Skip the bigram if each word does not start with a capital letter or a stop word.
            if not checkCapitalConsistency(bigram):
                continue

            # Determine whether this is purely a bigram or a part of trigram
            isTrigram = False
            try:
                # Concatenate the prominent trigram.
                trigrams = triDict[(bigram[0], bigram[1])]
                for word in trigrams:
                    trigram, frequency = (bigram[0], bigram[1], word), trigrams[word]                    
                    if checkCapitalConsistency(trigram) and (frequency >= 10):
                        # Concatenate the consecutive words of capital terms.                        
                        mergedNgrams['_'.join(trigram)] = frequency
                        outputFile.write('  - Concatenated trigram %s (%d)\n' % (trigram, frequency))
                        isTrigram = True
            except:
                # Note that if a sentence ends with the current bigrams, there could be no such key in trigram.
                pass
            
            # Concatenate the prominent bigram.
            if not isTrigram:
                frequency = biTfs[bigram]
                if frequency >= 100:
                    mergedNgrams['_'.join(bigram)] = frequency
                    outputFile.write('  - Concatenated bigram %s (%d)\n' % (bigram, frequency))
            
    pickle.dump(mergedNgrams, open(ResultsPath + '%s_capitalNgrams.dict' % dataset, 'wb'))


def mergeCapitalNgrams_batch(datasets=getDataFolders()):
    # For each dataset,
    for dataset in datasets:
        print('+ For the industry [%s]...\n' % dataset)
        mergeCapitalNgrams(dataset)
        

## Function:
##
"""
# Count term-frequencies and document-frequencies of unigrams and bigrams.

    
"""
def createCapitalNgrams(dataset):
    print('+ Reading the industry corpus for [%s]...' % dataset)
    dataFilenames = getDataFilenames(dataset)
    for report in ['AR', 'CSR']:
        filenames = sorted(list(dataFilenames[report]['all']))[0:]
        numDocuments = len(filenames)
        
        namename = dataset +'_'+report

        countNgrams(filenames,namename)
    mergeCapitalNgrams(dataset)
        





def replaceCapitalNgrams(dataset):
    print('+ Reading the industry corpus for [%s]...' % dataset)
    dataFilenames = getDataFilenames(dataset)
    for report in ['AR', 'CSR']:
        filenames = sorted(list(dataFilenames[report]['all']))[0:]
        numDocuments = len(filenames)
        
        namename = dataset +'_'+report

        #countNgrams(filenames,namename)
        #mergeCapitalNgrams(dataset)


        for (n, filename) in enumerate(filenames):
            # Read the document as a list of tokens.
            print("    - Working on %d/%d documents!" % (n+1, numDocuments), end='\r')
            
            


            mergedFile = open(ResultsPath + '%s_capitalNgrams.dict' % dataset, 'rb')
            mergedNgrams = pickle.load(mergedFile)    
            replacedTokens = nltk.word_tokenize(open(filename, errors ='ignore').read())
            
            # Repeat until no change,
            noUpdate = False
            while (not noUpdate):             
                tokens = list(replacedTokens)
                replacedTokens.clear()
                noUpdate = True

                # For each token,
                i, numTokens = 0, len(tokens)    
                while i < numTokens:
                    token = tokens[i]
                    #print('%d / %d' % (i, numTokens))
                    if token[0].isupper() & numTokens-i>1:
                        # Find the last index until which capital words continue.
                                              
                        for j in range(i+1, numTokens):
                            if j-i==0:
                                break  
                        #for j in range(i+1, numTokens):
                            if tokens[j][0].islower():
                                break

                        # If that is a trigram
                        if j >= i+3:
                            # Search every trigram.
                            triFreqs = [mergedNgrams.get('_'.join(tokens[k:k+3]), -1) for k in range(i, j)]
                            
                            # Pick the trigram that has the highest frequency.
                            maxIndex = triFreqs.index(max(triFreqs))
                            if triFreqs[maxIndex] >= 0:
                                # Insert.
                                k = i + maxIndex
                                replacedTokens.extend(tokens[i:k])
                                replacedTokens.append('_'.join(tokens[k:k+3]))
                                """
                                print(replacedTokens[-3:])
                                if replacedTokens[-2:][0] == 'entire' and replacedTokens[-2:][1] == 'Annual_Report':
                                    print('stop')                                    
                                """
                                replacedTokens.extend(tokens[k+3:j])                                
                                noUpdate = False
                            # Check the bigram.
                            else:
                                # Search every trigram.
                                biFreqs = [mergedNgrams.get('_'.join(tokens[k:k+2]), -1) for k in range(i, j)]
                                
                                # Pick the trigram that has the highest frequency.
                                maxIndex = biFreqs.index(max(biFreqs))
                                if biFreqs[maxIndex] >= 0:
                                    # Insert.
                                    k = i + maxIndex
                                    replacedTokens.extend(tokens[i:k])
                                    replacedTokens.append('_'.join(tokens[k:k+2]))
                                    """
                                    print(replacedTokens[-2:])
                                    if replacedTokens[-2:][0] == 'entire' and replacedTokens[-2:][1] == 'Annual_Report':
                                        print('stop')
                                    """
                                    i = k + 2                      
                                    noUpdate = False
                                    continue
                                
                        
                        # If that is a bigram,
                        elif j == i:
                            bigram = '_'.join(tokens[i:j])
                            if bigram in mergedNgrams:
                                # Insert the merged one if exists.
                                replacedTokens.append(bigram)
                               ######################33 
                                print(replacedTokens[-2:])
                                if replacedTokens[-2:][0] == 'entire' and replacedTokens[-2:][1] == 'Annual_Report':
                                    print('stop')                                    
                                ##################################
                                noUpdate = False
                            else:
                                # Insert individually.
                                replacedTokens.extend(tokens[i:j])

                        else:
                            replacedTokens.append(token)

                        # Jump to maintain linear complexity.                       
                        i = j

                    else:
                        replacedTokens.append(token)
                        i += 1

            basename, extension = os.path.splitext(filename)
            with open(basename + '_new' + extension, 'w') as newFile:
                newFile.write(' '.join(replacedTokens))


def extractVocabSets(dataset):
    uniTfs = {}
    uniDfs = {}
   

  
    uniTfs['AR']  = pickle.load(open(ResultsPath + '%s_%s_unigram-tfs.Counter' % (dataset, 'AR'), 'rb')) 
    uniDfs['AR']  = pickle.load(open(ResultsPath + '%s_%s_unigram-dfs.Counter' % (dataset, 'AR'), 'rb'))
    uniTfs['CSR'] = pickle.load(open(ResultsPath + '%s_%s_unigram-tfs.Counter' % (dataset, 'CSR'), 'rb')) 
    uniDfs['CSR'] = pickle.load(open(ResultsPath + '%s_%s_unigram-dfs.Counter' % (dataset, 'CSR'), 'rb'))

    

    
    print('hello')
    arTerms, csrTerms = set(uniTfs['AR'].keys()), set(uniTfs['CSR'].keys()) 
    commonTerms = arTerms.intersection(csrTerms)
    terms = {'AR':arTerms - commonTerms, 'CSR':csrTerms - commonTerms, 'COMMON':commonTerms}

    

    
    for terms in ['AR', 'CSR']:
        with open(ResultsPath + '%s_unigramVocabs_%s.txt' % (dataset, terms), 'w',encoding="ISO-8859-1") as outputFile:
            
            for unigram in sorted(uniTfs[terms], key=uniTfs[terms].get, reverse=True):
                outputFile.write('        %s (%d, %d)\n' % (unigram, uniTfs[terms][unigram], uniDfs[terms][unigram]))





def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts,dataset):
    MalletStopWords = set([word.strip() for word in open(MalletStopPath + dataset +'.txt', 'r',encoding="ISO-8859-1").readlines()])
    common_stops = []
    stop_words = stopwords.words('english')
    stop_words.extend(MalletStopWords)
    #stop_words.extend(common_stops)

    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.max_length = 5000000

def process_words(texts, stop_words,bigram_mod,trigram_mod,allowed_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    
    """Convert a document into a list of lowercase tokens, build bigrams-trigrams, implement lemmatization"""
    
    # remove stopwords, short tokens and letter accents 
    texts = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in texts]
    
    # bi-gram and tri-gram implementation
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    texts_out = []
    
    # implement lemmatization and filter out unwanted part of speech tags
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_tags])
    
    # remove stopwords and short tokens again after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in texts_out]    
    
    return texts_out

from gensim.models import TfidfModel
from openpyxl.workbook import Workbook
        

#
#
#
#
def createModel(dataset,numTopics):
    print('+ Reading the industry corpus for [%s]...' % dataset)
    dataFilenames = getDataFilenames(dataset)
    data_words = []
    fnames=[]
    for report in ['Reports']:
        filenames = sorted(list(dataFilenames[report]['all']))[0:]
        numDocuments = len(filenames)
       
        for (n, filename) in enumerate(filenames):
            #comment this out for stage 1
            fnames.append(filename)


            print("    - Working on %d/%d documents!" % (n+1, numDocuments), end='\r')

           #########################################################
            """with open(filename, 'r', encoding='ISO-8859-1') as file:


                

                temp_list =  list(sent_to_words(file))
                data_words = data_words + temp_list
    
    
        with open(dataset+"-TEXTS.txt", "w") as textfile:
            for item in data_words:
                    textfile.write("%s\n" % item)"""
    
    #################################################################
    

   
    """stop_words = nltk.corpus.stopwords.words('english')
    my_file = [word.strip() for word in open(dataset+"-TEXTS.txt", 'r',encoding="ISO-8859-1").readlines()]
    
    bigram = gensim.models.Phrases(my_file, min_count=20, threshold=50)
    trigram = gensim.models.Phrases(bigram[my_file], threshold=50)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)







    data_words.extend(my_file)

    data_words = process_words(data_words,stop_words,bigram_mod,trigram_mod)
    
    #data_proc = remove_stopwords(data_words,dataset)
                #print(data_words[:1][0][:50])
    with open(dataset+"-TEXTS2.txt", "w") as textfile:
            for item in data_words:
                    textfile.write("%s\n" % item)"""

    
    my_file = [word.strip() for word in open(dataset+"-TEXTS2.txt", 'r',encoding="ISO-8859-1").readlines()]
    data_words.extend(my_file)

   
    #create dictionary

    data_words = remove_stopwords(data_words,dataset)
    id2word = corpora.Dictionary(data_words)

    #create corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    #print(corpus[:1][0][:30])

    tfidf = TfidfModel(corpus,id2word=id2word)

    low_value = 0.05
    words=[]

    words_missing_in_tfidf=[]

    for i in range(0, len(corpus)):
        bow = corpus[i]
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        low_value_words = [id for id,value in tfidf[bow] if value < low_value]
        drops =low_value_words+words_missing_in_tfidf
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf =[id for id in bow_ids if id not in tfidf_ids]

        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        corpus[i] = new_bow

    """model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_words, start=20, limit=160, step=10)

    # Show graph
    limit=160; start=20; step=10
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig(dataset+'_COHERENCE.png')
    
    
   
    


    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
    with open(dataset+"-COHERENCE.txt", "w") as textfile:
            for m, cv in zip(x, coherence_values):
                    textfile.write("%d,%d\n" % (m, round(cv, 4)))"""

    
     # Build LDA model
    lda_model = gensim.models.LdaModel(
                        corpus=corpus,
                            num_topics=numTopics,
                        id2word=id2word)
    # Print the Keyword in the 10 topics
    #pprint(lda_model.show_topics(formatted=False))

    # Compute Coherence Score
    #coherence_model_ldamallet = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
    #coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    #print('Coherence Score: ', coherence_ldamallet)

    #save file in pickle
    #pickle.dump(lda_model, open(ResultsPath, "wb"))

    #(distribution of topics for each document)
    tm_results = lda_model[corpus]

    

    #You can get top 20 significant terms and their probabilities for each topic as below:

    


 #lda_model.show_topics(n, topn=20)

    topics = ([(term, round(wt, 3)) for term, wt in lda_model.show_topic(n, topn=20)] for n in range(0, lda_model.num_topics))
    

    #mds = 'pcoa'
    topic_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    #topic_data = pyLDAvis.gensim_models._extract_data(lda_model, corpus, id2word)
    topic_data

    
    all_topics = {}
    num_terms = 20 # Adjust number of words to represent each topic
    lambd = 0.4 # Adjust this accordingly based on tuning above
    for i in range(0,numTopics+1): #Adjust this to reflect number of topics chosen for final LDA model
        topic = topic_data.topic_info[topic_data.topic_info.Category == 'Topic'+str(i)].copy()
        topic['relevance'] = topic['loglift']*(1-lambd)+topic['logprob']*lambd
        all_topics['Topic '+str(i)] = topic.sort_values(by='relevance', ascending=False).Term[:num_terms].values
    #pd.DataFrame(all_topics).T
    values = all_topics.values()


    
    
    values_list = list(values)
    listerine = []
   
    for sentence in values_list[1:]:
        listerine.append(sentence)

    gen_list = list(topics)   
    



    #topics = ((term, round(wt, 3)) for term, wt in all_topics) 
    res = [tuple for x in listerine for tuple in gen_list if tuple[0] == x]
    

    #list of topics per document
    corpus_topics = [listerine for listerine in tm_results]
   
    

    #create dataframe
    # set column width
    pd.set_option('display.max_colwidth', -1)
    #topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics],
    try:
        topics_df = pd.DataFrame([", ".join([term for term in line])for line in listerine],
        columns = ['Terms per Topic'],
        index=['Topic'+str(t) for t in range(1, len(listerine)+1)] )
        topics_df
        topics_df.to_csv(dataset+'-TOPICS.csv')
    except Exception:
        pass

    # create a dataframe
    corpus_topic_df = pd.DataFrame()

    # get the Titles from the original dataframe
    
        
        
    
    d=0
    try:
        for item in corpus_topics:
                temp_df = pd.DataFrame()
                
                ffname = fnames[d]

                temp_df['Title'] = ffname
                temp_df['Topic'] =  [item[0]+1 for item in item]
                temp_df['Contribution %'] = [item[1]*100 for item in item]
                temp_df['Topic Terms'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in item]
                indexes =len(temp_df)

                for i in range(0,indexes):
                    temp_df.at[i,'Title']=ffname
                d=d+1
                
            
                corpus_topic_df=corpus_topic_df.append(temp_df)
                corpus_topic_df.head()
                temp_df=temp_df = pd.DataFrame()
    except Exception:
        pass        
    corpus_topic_df.to_csv(dataset+'.csv')
   
        #document counts per topic and its percentage in the corpus
        #dominant_topic_df = corpus_topic_df.groupby('Dominant Topic').agg(
         #                   Doc_Count = ('Dominant Topic', np.size),
         #                   Total_Docs_Perc = ('Dominant Topic', np.size)).reset_index()

        #dominant_topic_df['Total_Docs_Perc'] = dominant_topic_df['Total_Docs_Perc'].apply(lambda row: round((row*100) / len(corpus), 2))

        #dominant_topic_df"""
        
                
def cleanIndustry(dataset):
    createCapitalNgrams(dataset)
    replaceCapitalNgrams(dataset)
    extractVocabSets(dataset)
              
 #find the coherence values for each industry               
def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    
    """Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics"""
    
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        lda_model = gensim.models.LdaModel(
                        corpus=corpus,
                            num_topics=num_topics,
                        id2word=dictionary)
        model_list.append(lda_model)
        coherencemodel = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
              
GlobalStopWords = ['num', 'NUM', 'Num', 'gNUM', 'NUMkh']



createModel('IT CONSULTING',120)
createModel('IT COMMUNICATION',70)
createModel('INDUSTRIAL ELECTRONICS',50)
createModel('CONSUMER FINANCE',120)
createModel('AEROSPACE DEFENSE', 50)

    


"""if __name__ == '__main__':


   
    #createModel('INDUSTRIAL CONGLOMARATE')
    #createModel('INDUSTRIAL ELECTRONICS')

        createModel('PAPER PACKAGING',90)
        createModel('PHARMACEUTICALS',90)
        createModel('RAILROAD',150)
        createModel('REAL ESTATE',150)
        createModel('REGIONAL BANKS',90)
        createModel('RESTAURANTS',70)
        createModel('RETAILERS',80)
        createModel('SEMICONDUCTORS',60)
        createModel('SIN INDUSTRY',120)
        createModel('SOFTWARE',90)

    createModel('AEROSPACE DEFENSE', 50)
    createModel('AIRLINE AND FREIGHT',50)
    createModel('APPAREL',80)
    createModel('ASSET MANAGEMENT',120)
    createModel('AUTOMOTIVE',110)
    createModel('BIOTECHNOLOGY',70)
    createModel('BUILDING CONSTRUCTION',70)
    createModel('CHEMICALS',50)
    createModel('COMMUNICATION',60)
    createModel('CONSUMER FINANCE',120)
    createModel('DEPARTMENT GENERAL STORE',140)
    createModel('ELECTRIC UTILITIES',100)
    createModel('ENERGY EQUIPMENT AND SERVICES',110)
    createModel('ENERGY PRODUCTION', 60)
    createModel('FOREX',90)
    createModel('HARDWARE',90)
    createModel('HEALTHCARE DISTRIBUTORS',80)
    createModel('HEALTHCARE EQUIPMENT',100)
    createModel('HEALTHCARE FACILITIES',110)
    createModel('HOTEL CASINO',90)
    createModel('HOUSE HOME',90)
    createModel('HOUSEHOLD PERSONAL CARE',130)
    createModel('INDUSTRIAL CONGLOMARATE',110)
    createModel('INDUSTRIAL ELECTRONICS',50)
    createModel('INDUSTRIAL MACHINERY',50)
    createModel('INDUSTRIAL SUPPORT SERVICES',140)
    createModel('INSURANCE',60)
    createModel('INTERNET',150)
    createModel('INVESTMENT',50)
    createModel('IT COMMUNICATION',70)
    
    createModel('MEDIA',140)
    createModel('MULTI UTILITIES',130)
    createModel('NATURAL MATERIALS',120)
    createModel('OTHER MATERIALS',130)
    createModel('PACKAGED GOODS',80)


    
    createModel('PAPER PACKAGING',90)
    createModel('PHARMACEUTICALS',90)
    createModel('RAILROAD',150)
    createModel('REAL ESTATE',150)
    createModel('REGIONAL BANKS',90)
    createModel('RESTAURANTS',70)
    createModel('RETAILERS',80)
    createModel('SEMICONDUCTORS',60)
    createModel('SIN INDUSTRY',120)
    createModel('SOFTWARE',90)"""
    
    
    

    
   
    
    ## Default stopwords.

