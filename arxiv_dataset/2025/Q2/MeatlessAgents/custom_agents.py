import json
import logging
from enum import Enum
from pydantic import create_model
from typing import Sequence, List

from tqdm.auto import tqdm

from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_core.models._types import UserMessage

from autogen_agentchat.base import Response

from autogen_agentchat.messages import (
    ChatMessage,
    HandoffMessage,
    StopMessage,
    TextMessage,
)
from autogen_core.models import (
    ChatCompletionClient,
    UserMessage,
)
from autogen_agentchat.agents._base_chat_agent import BaseChatAgent

from autogen_agentchat import EVENT_LOGGER_NAME
event_logger = logging.getLogger(EVENT_LOGGER_NAME)

class MyAgent(BaseChatAgent):
    def __init__(
            self,
            name: str,
            model_client: ChatCompletionClient,
            description: str,
            profile: str,
            reflection_task: str,
            response_task: str,
            pbar: tqdm,
            questionnaire_data: list | None = None,
            questionnaire_repetitions: int = 10,
            # TODO add a semi-constant seed!!
    ) -> None:
        super().__init__(name=name, description=description)
        self._model_client = model_client
        self._messsage_history = []
        self._thoughts = []
        self._questionnaire_responses = []
        self._profile = profile + '\n'
        self._reflection_instr = reflection_task
        self._response_instr = response_task.format(name=name, profile=profile) + '\n'
        self._questionnaire_data = questionnaire_data
        self._questionnaire_repetitions = questionnaire_repetitions
        self._pbar = pbar
        self._round = 1 # how long has the conversation been ongoing?
    
    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        return
    
    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:

        for msg in messages:
            if msg.source != "user": # exclude the opening task
                self._messsage_history.append(msg)
        
        ## Simulate the reflection
        print('round', self._round, self._name, 'thinking...')
        input_msg = self._reflection_instr.format(name = self.name,
                                                  profile = self._profile,
                                                  transcript = self._create_transcript(self._messsage_history[-5:])) #TODO: adjust maximal history
        reflection = await self._model_client.create([UserMessage(content=input_msg, source='user')],
                                                 cancellation_token=cancellation_token)
        self._thoughts.append(reflection)
        
        ## Fill out the questionnaire, if provided
        if self._questionnaire_data is not None:
            print('round', self._round, self._name, 'filling out the questionnaire...')
            questionnaire_prefix = f'You are {self._name}.\n{self._profile}'
            questionnaire_prefix += self._create_transcript(self._messsage_history[-5:])
            questionnaire_prefix += f'YOUR REFLECTION: {reflection.content}\n'

            #questionnaire_response = await self._model_client.create([UserMessage(content=input_msg, source='user')],
            #                                        cancellation_token=cancellation_token)
            for section in self._questionnaire_data:
    
                ## Construct the local text context from the JSON
                input_msg = questionnaire_prefix
                if 'introduction' in section.keys():
                    input_msg += section['introduction'] + '\n'
                else:
                    input_msg += 'Please respond to the following questionnaire:\n'
                input_msg += '\n'.join([question['id'] + ': ' + question['text'] for question in section['questions']]) + '\n'

                # open ended responses should be unstructured
                if 'options' in section['questions'][0]:
                    input_msg += (
                        'Answer in the following JSON format: {' + 
                        ', '.join([f"'{question['id']}': your response" for question in section['questions']]) + 
                        '}'
                    )
                    input_msg += (
                        '\nWhere your response is from one of the following options: ' +
                        str(section['questions'][0]['options']) + 
                        '\nDo not provide any additional explanation, just one of the possible answer options.'
                    )

                ## Construct the dynamic pydantic model
                # This code does not produce duplicate enums after conversion
                DynamicEnums = [Enum('DynamicEnum', {option: option for option in question['options']})
                                if 'options' in question else str # handle questions without answer options as open ended
                                for question in section['questions']]
                q_ids = [question['id'] for question in section['questions']]
                answer_types = {q_id: (enum, ...) for q_id, enum in zip(q_ids, DynamicEnums)}
                #answer_types.update({'explanation': (str, ...)}) # OPTIONAL: add an additional explanation after each section
                DynamicAnswer = create_model("DynamicAnswer", **answer_types)

                # TODO: parallelize this!
                for repetition in range(self._questionnaire_repetitions):

                    # only use structured outputs with closed-ended questions
                    if 'options' in section['questions'][0]:
                        extra_create_args = {"response_format": DynamicAnswer}
                    else:
                        extra_create_args = {}

                    response = await self._model_client.create(
                        [UserMessage(content=input_msg, source='user')],
                        extra_create_args = extra_create_args
                        )
                    
                    # parse structured response to closed-ended questions
                    if 'options' in section['questions'][0]:
                        result = json.loads(response.content)
                    else:
                        result = {'11.2': response.content} # TODO: fix this to be the correct question id
                    result['round'] = self._round
                    result['repetition'] = repetition
                    self._questionnaire_responses.append(result)

        ## Create a response
        print('round', self._round, self._name, 'creating answer...')
        input_msg = self._response_instr + self._create_transcript(self._messsage_history[-5:])
        input_msg += f'YOUR REFLECTION: {reflection.content}'
        #print(f'---------- {self._name} reflection ----------')
        #print(input_msg)
        result = await self._model_client.create([UserMessage(content=input_msg, source='user')],
                                                 cancellation_token=cancellation_token)
        response = Response(
            chat_message=TextMessage(content=result.content, source=self.name, models_usage=result.usage),
            inner_messages=[],
        )
        self._messsage_history.append(response.chat_message)
        self._round += 1
        self._pbar.update(1)

        return response
    
    def _create_transcript(self, messages: Sequence[ChatMessage]) -> str:
        transcript = "THE CONVERSATION SO FAR:\n"
        for message in messages:
            if isinstance(message, TextMessage | StopMessage | HandoffMessage):
                transcript += f"{message.source}: {message.content}\n"
            else:
                raise ValueError(f"Unexpected message type: {message} in {self.__class__.__name__}")
        return transcript

    async def _create_opener(self, task) -> TextMessage:
        print(self._name, 'creating opening statement...')
        task_msg = self._profile + task
        result = await self._model_client.create([UserMessage(content=task_msg, source='user')])
        return TextMessage(content=result.content, source=self.name, models_usage=result.usage)
