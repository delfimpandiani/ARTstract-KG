





# this is the folk values detector for the ISWS 2023 Ravenclaw team.
# Updated at 07.09.2023


from distutils.command.build import build
import requests
import rdflib
from rdflib import Graph, Literal, RDF, URIRef, Namespace, BNode
import SPARQLWrapper
from SPARQLWrapper import SPARQLWrapper, JSON, N3, TURTLE, RDF, CSV
import json
import os
from collections import defaultdict
import pandas as pd
import csv
from urllib.error import HTTPError
import time


# Here you specify the files from which you will be checking the activation of some value, emotion, etc.
BEfile = "trigger_sources/BE_emotion_dictionary.csv"
MFTfile = "trigger_sources/mftriggers_dictionary.csv"
BHVfile = 'trigger_sources/bhv_dictionary.csv'
FOLKfile = 'trigger_sources/taf_dictionary.csv'


haidt = URIRef('https://w3id.org/spice/SON/HaidtValues#')
valuetriggers = URIRef('http://www.ontologydesignpatterns.org/ont/values/valuecore_with_value_frames.owl#triggers')
emotriggers = URIRef('http://www.ontologydesignpatterns.org/ont/emotions/EmoCore.owl#triggers')
graphFor = URIRef('https://w3id.org/sdg/meta#graphFor')





# Here you use the api keys for FRED. Use the already included Bearer key.
headers = {
    'accept': 'text/turtle',
    'Authorization': 'Bearer b67a0577-8c76-3889-89b2-cf3dceab4a0e',
}



# Here you introduces properties to generate new triples on the graph
#valuetriggers = URIRef('http://www.ontologydesignpatterns.org/ont/values/valuecore_with_value_frames.owl#triggers')
#graphFor = URIRef('https://w3id.org/sdg/meta#graphFor')


# Here you specify the path for your csv with sentences to be analyzed. Include the sep="\t" if using a .tsv file
doc = pd.read_csv('output.tsv', sep="\t") #, sep="\t")


# Here you introduce a variable which will be used to avoid timeout in FRED calls
t = 20


# This function is to write the output in a new file
def scrittura_file(df, path, nome_file, header):
    if not os.path.isfile(nome_file):
        df.to_csv(nome_file, index=False, header=header) #, sep="\t"
    else:
        df.to_csv(nome_file, mode='a', index=False, header=False) #, sep="\t"



# This function is to create (and return) a dictionary out of the desired csv file, in this script are Values, but it could be anything
def build_dict(file):
    with open(file) as ttl_file:
        input_file = csv.DictReader(open(file))
        dict = {row['s']:row['o'] for row in input_file}
        return dict





# This function is to retrieve value triggers: it generates a graph from FRED, and then it takes the ttl and, 
# for each s,p,o checks if some s or o is a value trigger, 
# then it returns as output a ttl with triples about values activation.
def find_trigs(doc):
    i = 1
    for index,row in doc.iterrows():
        try:
            g = rdflib.Graph()

            obj = []
            usefulobj = []
            synsetlist = []
            # Here you introduce a list to store the retrieved frames
            framelist = []
            # declare the header of the column containing the text you want to be passed to FRED as variable "txt"
            values_list = []
            value_trigs_list = []
            emotions_list = []
            emo_trigs_list = []
            valuedict = build_dict(MFTfile)
            emodict = build_dict(BEfile)
            bhvdict = build_dict(BHVfile)
            folkdict = build_dict(FOLKfile)
            obj_value_graph = rdflib.Graph()
            obj_emo_graph = rdflib.Graph()
            txt = row['caption']
            image_id = row['image_id']
            print(txt)
            print(image_id)


            # Leave these parameters as they are
            params = (
                ('text', txt),
                ('wfd_profile', 'b'),
                ('textannotation', 'earmark'),
                ('wfd', True),
                ('roles', False),
                ('alignToFramester', True),
                ('semantic-subgraph', True)
                )

            # Here you actually make a call to FRED
            response = requests.get('http://wit.istc.cnr.it/stlab-tools/fred', headers=headers, params=params)
            fredg = g.parse(data=response.text, format='ttl')

            #template = ('https://template/sdg/graph_'+str(i))
            
            # Here you keep track of the sentence generating the graph in the graph itself, could be useful
            #fredg.add((URIRef(template), graphFor, Literal(txt)))


            # Here you iterate only on some specific leaf nodes of the graph, since not all the nodes can be semantic triggers
            for s,p,o in fredg:
                s = str(s).replace("http://www.w3.org/2006/03/wn/wn30/instances", "https://w3id.org/framester/wn/wn30/instances")
                o = str(o).replace("http://www.w3.org/2006/03/wn/wn30/instances", "https://w3id.org/framester/wn/wn30/instances")
                s = str(s).replace("http://www.ontologydesignpatterns.org/ont/vn/data", "https://w3id.org/framester/vn/vn31/data")
                o = str(o).replace("http://www.ontologydesignpatterns.org/ont/vn/data", "https://w3id.org/framester/vn/vn31/data")

                obj.append(s)
                obj.append(o)

                # 'https://w3id.org/framester/vn/vn31/data/Attack_33000000'
                # print(obj)

                # Here we filter only those nodes on the graph that could be triggers for some emotion or value
                for x in obj:
                    if 'dbpedia' in str(x):
                        usefulobj.append(x)
                    if 'synset' in str(x):
                        usefulobj.append(x)
                        synsetlist.append(x)
                    if 'framestercore' in str(x):
                        usefulobj.append(x)
                    if 'vn31' in str(x):
                        usefulobj.append(x)
                    if 'pbdata' in str(x):
                        usefulobj.append(x)


                for n in set(usefulobj):
                    for valuekey1 in valuedict.keys():
                        if n == valuekey1:
                            obj_value_graph.add((URIRef(n),valuetriggers,URIRef(valuedict[valuekey1])))
                    for valuekey2 in bhvdict.keys():
                        if n == valuekey2:
                            obj_value_graph.add((URIRef(n),valuetriggers,URIRef(bhvdict[valuekey2])))
                    for valuekey3 in folkdict.keys():
                        if n == valuekey3:
                            obj_value_graph.add((URIRef(n),valuetriggers,URIRef(folkdict[valuekey3])))
                    for emokey in emodict.keys():
                        if n == emokey:
                            obj_emo_graph.add((URIRef(n),emotriggers,URIRef(emodict[emokey])))
                    
                finalg = fredg + obj_value_graph + obj_emo_graph

                for s,p,o in finalg:
                    if valuetriggers in p:
                        values_list.append(o)
                        value_trigs_list.append(s)
                    if emotriggers in p:
                        emotions_list.append(o)
                        emo_trigs_list.append(s)
                for s, p, o in fredg:
                    if 'framestercore' in str(o):
                        framelist.append(o)

            # Here you are generating and saving a Turtle file for each sentence you are passing to FRED, 
            # and it is going be numbered starting from the "i" variable declared at the beginning
            #fredg.serialize(destination=('/Users/sdg/Desktop/GeometryOfMeaning/MusicBoSituations/'+str(i)+"_GRAPH.ttl"))

            # Here you are preparing to build a json file as output, including your value activations
            txt = row['caption']
            frame_set = set(o.replace('https://w3id.org/framester/data/framestercore/', 'frame:').strip() for o in framelist)
            values_set = set(v.replace('https://w3id.org/spice/SON/HaidtValues#','mft:').replace('https://w3id.org/spice/SON/SchwartzValues#','bhv:').replace('http://www.ontologydesignpatterns.org/ont/values/FolkValues.owl#','folk:').strip() for v in values_list)
            values_trigs_set = set(value_trigs_list)
            emo_set = set(e.replace('http://www.ontologydesignpatterns.org/ont/emotions/BasicEmotions.owl#','be:').strip() for e in emotions_list)
            emo_trigs_set = set(emo_trigs_list)
            values_trigs_set.update(emo_trigs_set)
            values_trigs_set.update(synsetlist)
            print(values_trigs_set)
            # Here you build the json, include keys and fields from your original file that you want to keep in the json output
            out = {
                'image_id': [image_id],
                'image description':[txt],
                'frames':[', '.join(frame_set)],
                'Emotions':[', '.join(emo_set)],
                'Values':[', '.join(values_set)],
                'Triggers': [', '.join(values_trigs_set)]
            }

            
            # Here you check advancements in your script
            print(f"{index}/{len(doc)}")
            print(out)

            # Here you write the csv in an output file
            df = pd.DataFrame(out)
            scrittura_file (df, '', 'all_frames_n_triggers.csv', [k for k in out.keys()])

    
        except Exception:
            print("Exception")
            time.sleep(t)

        i = i+1



find_trigs(doc)





