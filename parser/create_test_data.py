from message_parser import *
import pickle

conversations = parse_xml("pan12-sexual-predator-identification-test-corpus-2012-05-17.xml")
print("Parsing inputs...\n")
inputs, authors = get_documents(conversations)
pickle.dump(inputs, open("test_inputs_dict.p", "wb"))
print("Parsing outputs...\n")
outputs, predators = get_outputs_for_authors("pan12-sexual-predator-identification-groundtruth-problem1.txt", authors)
pickle.dump(outputs, open("test_outputs_dict.p", "wb"))
lines = []
file = open("pan12-sexual-predator-identification-groundtruth-problem2.txt","r")
for line in file:
    line = line.split()
    lines.append((line[0], line[1]))
pickle.dump(lines, open("lines.p", "wb"))