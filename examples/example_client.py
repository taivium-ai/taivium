# Example usage of PrivacyClient
import os
from typing import Any, Dict, List, Optional

from taivium.client import PrivacyClient

# pylint: disable=invalid-name

EXAMPLE_MESSAGES: List[Dict[str, str]] = [
    {"role": "user", "content": "John Doe's phone number is 123-456-7890."}
]


def run_basic_client_example(
    client: Optional[PrivacyClient] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    model: str = "gpt-4o",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the basic PrivacyClient usage example.

    Demonstrates:
    - Creating a :class:`~taivium.client.PrivacyClient` (or accepting an
      injected one for testing)
    - Making a de-identified chat completion call with ``deid_response=True``
    - Inspecting the session mapping to see which entities were anonymised
    - Resetting the session

    Args:
        client: Optional pre-constructed :class:`PrivacyClient`.  When ``None``
                (default), a new client is created using the ``OPENAI_KEY``
                environment variable.
        messages: Input message list in OpenAI format.
                  Defaults to :data:`EXAMPLE_MESSAGES`.
        model: OpenAI model identifier forwarded to the completions endpoint.
               Defaults to ``"gpt-4o"``.
        verbose: When ``True``, prints the response and session mapping to stdout.

    Raises:
        EnvironmentError: When *client* is ``None`` and ``OPENAI_KEY`` is unset.

    Returns:
        dict with keys:

        ``response``
            The ``ChatCompletion`` object returned by the LLM (entity tokens in
            the message content are replaced with original values because
            ``deid_response=True``).

        ``session_mapping``
            Snapshot of the entity mapping captured *before* the session is
            reset (keyed by token ID).

    Example output (abridged)::

        De-identified Response:
        - John Doe will be contacted at 123-456-7890.

        Session Mapping:
        - Token: PERSON_abc123, Original: John Doe
        - Token: PHONE_def456, Original: 123-456-7890
    """
    if messages is None:
        messages = EXAMPLE_MESSAGES

    if client is None:
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise EnvironmentError("The environment variable OPENAI_KEY is not set.")
        client = PrivacyClient(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        deid_response=True,
    )

    if verbose:
        print("\nDe-identified Response:")
        for choice in response.choices:
            print(f"- {choice.message.content}")

        print("\nSession Mapping:")
        for token, mapping in client.session_mapping.items():
            original = mapping.get("text", "N/A")
            print(f"- Token: {token}, Original: {original}")

    result = {
        "response": response,
        "session_mapping": dict(client.session_mapping),
    }

    client.reset_session()
    return result


if __name__ == "__main__":
    run_basic_client_example()
