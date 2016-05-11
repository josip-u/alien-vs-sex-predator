import xml.dom.minidom


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
            if not message.hasAttribute("text"):
                continue
            text = message.getElementsByTagName("text")[0].childNodes[0].data
            inputs.append(text)
            if (id, line) in positive_inputs:
                outputs.append(1)
            else:
                outputs.append(0)
    return inputs, outputs
