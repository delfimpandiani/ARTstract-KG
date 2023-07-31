import json
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

# Load the JSON data
with open("merged_data.json", "r") as file1:
    data_dict = json.load(file1)

# Create a new RDF graph
g = Graph()

# Define the namespaces
art_ns = Namespace("http://example.org/artstract/")
conceptnet_ns = Namespace("http://example.org/conceptnet/")

# (The rest of the code remains the same as in the previous example)

# Serialize the graph to Turtle format
turtle_data = g.serialize(format="turtle")

# Print the Turtle RDF data
print(turtle_data)

# Save the Turtle RDF data to a file
with open("turtle_data.ttl", "w") as outfile:  # Open in regular text mode (not binary mode)
    outfile.write(turtle_data)
