import time

import openai


def wait_for_server_ready(client, model, seconds_to_wait: int):
    found = False
    i = 0
    seconds_waited_so_far = 0
    seconds_to_wait_at_error = 1
    while not found and seconds_waited_so_far < seconds_to_wait:
        try:
            print(
                f"Trying to connect to server, waited {seconds_waited_so_far}s so far."
            )
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "dummy"},
                    {"role": "user", "content": "dummy2"},
                ],
            )
            found = True
        except openai.APIConnectionError as e:
            found = False
            print(
                f"Server was not ready and got error {str(e)}, sleeping for {seconds_to_wait_at_error} seconds"
            )
            time.sleep(seconds_to_wait_at_error)
            seconds_waited_so_far += seconds_to_wait_at_error
            seconds_to_wait_at_error *= 2
        i += 1
    if not found:
        raise ValueError(
            f"Server could not be reached after waiting {seconds_waited_so_far}s."
        )
