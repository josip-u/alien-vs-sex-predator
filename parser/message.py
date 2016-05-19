from enum import Enum

class Time(Enum):
	morning = 1
	day = 2
	night = 3


class Message:

	def __init__(self, text, time, conversation_id, message_line):
		"""
		:param	text:				string - contains message text
		:param	time:				int - from range(0, 23) - hour when message was sent
		:param	conversation_id:	string - conversation id
		:param	message_line:		string - number of this message in its conversation
		"""
		self._text = text
		self._time = time
		self._conversation_id = conversation_id
		self._message_line = message_line