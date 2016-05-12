from message_parser import *
import pickle

conversations = parse_xml("pan12-sexual-predator-identification-training-corpus-2012-05-01.xml")
inputs, authors = get_documents(conversations)
outputs, predators = get_outputs_for_authors("pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt", authors)
pickle.dump(inputs, open("inputs_dict.p", "wb"))
pickle.dump(outputs, open("outputs_dict.p", "wb"))