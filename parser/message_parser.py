import xml.dom.minidom
from html import unescape
from message import Message


def parse_xml(file_name):
    """
    :param file_name:   string
    :return:            list - list of conversations
        - returns conversations from given xml file
    """
    domtree = xml.dom.minidom.parse(file_name)
    collection = domtree.documentElement
    conversations = collection.getElementsByTagName("conversation")
    return conversations


def get_positive_inputs(file_name):
    """
    :param file_name:   string
    :return:            list of tuples
        - returns a list of tuples (conversation_id, message_id) for
        messages which are the most distinctive of the predator bad
        behavior
        - used for getting expected output for machine learning
    """
    positive_inputs = []
    file = open(file_name, "r")
    for line in file:
        line = line.split()
        positive_inputs.append((line[0], line[1]))
    file.close()
    return positive_inputs


def parse_messages(conversations, positive_inputs):
    """
    :param conversations:       list
    :param positive_inputs:     list of tuples
    :return:                    tuple (list)
        - returns a tuple of lists, (input, output), where input is a list of strings (messages) and
        output is a list of integers (0 or 1) where 1 indicates positive class and 0 negative
    """
    inputs = []
    outputs = []
    for conversation in conversations:
        id = conversation.getAttribute("id")
        messages = conversation.getElementsByTagName("message")[:]
        for message in messages:
            line = message.getAttribute("line")
            node = message.getElementsByTagName("text")
            if not node or not node[0].childNodes:
                continue
            text = message.getElementsByTagName("text")[0].childNodes[0].data
            text = unescape(text)
            inputs.append(text)
            if (id, line) in positive_inputs:
                outputs.append(1)
                covered_positive_inputs.append((id, line))
            else:
                outputs.append(0)
    return inputs, outputs
	
	
def get_documents(conversations):
    """
    :param conversations:       list - DOM type
    :return:                    dictionary, set
        - returns a dictionary of documents by author
        - document is a list of Message class objects
        - each message includes a message text, time stamp, conversation id and message line
        - also returns a set of authors' id
    """
    inputs = {}
    for conversation in conversations:
        conversation_id = conversation.getAttribute("id")
        messages = conversation.getElementsByTagName("message")[:]
        for message in messages:
            message_line = message.getAttribute("line")
            text_node = message.getElementsByTagName("text")
            author_node = message.getElementsByTagName("author")
            time_node = message.getElementsByTagName("time")
            if not (text_node and text_node[0].childNodes and author_node
                    and author_node[0].childNodes):
                continue
            text = text_node[0].childNodes[0].data
            author = author_node[0].childNodes[0].data
            time = time_node[0].childNodes[0].data
            text = unescape(text)
            time = int(time[:2])
            message_obj = Message(text, time, conversation_id, message_line)
            if author not in inputs.keys():
                inputs[author] = [message_obj]
            else:
                inputs[author].append(message_obj)
    return inputs, set(inputs.keys())
	

def get_outputs_for_authors(file_name, authors):
    """
    :param file_name: string
    :param authors:   set
    :return:          dictionary
    - returns a dictionary of outputs for ML
    - file_name is the name of the file with predators' id
    - authors is the set of all users' id
    """
    predators = []
    outputs = {}
    for author in authors:
        outputs[author] = 0
    file = open(file_name, "r")
    for line in file:
        line = line.split()
        line = line[0]
        outputs[line] = 1
        predators.append(line)
    return outputs, predators
