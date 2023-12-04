import backoff
import openai
from functools import lru_cache


# @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, 
#             openai.error.APIError, openai.error.APIConnectionError, 
#             openai.error.Timeout, openai.error.ServiceUnavailableError))
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(
        model=kwargs['engine'],
        temperature=kwargs['temperature'], 
        max_tokens=kwargs['max_tokens'], 
        presence_penalty=kwargs['presence_penalty'], 
        frequency_penalty=kwargs['frequency_penalty'], 
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": kwargs['prompt']},
            ]
        )

@lru_cache(maxsize=10000)
def get_gpt_output(prompt, gpt_logger=None, **kwargs):
    response = completions_with_backoff(prompt=prompt, engine=kwargs['engine'], \
                                        temperature=kwargs['temperature'], max_tokens=kwargs['max_tokens'], \
                                        presence_penalty=kwargs['presence_penalty'], frequency_penalty=kwargs['frequency_penalty'])

    response_str = response['choices'][0]['message']['content']
    if gpt_logger is not None:
        gpt_logger.write(prompt)
        gpt_logger.write(response_str)
        gpt_logger.write('#' * 55)
    return response_str