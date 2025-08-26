from typing import Any, List, Optional, Type

from fastapi import HTTPException
from litellm import acompletion
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    InvalidRequestError,
    OpenAIError,
    RateLimitError,
    ServiceUnavailableError,
)
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, wait_exponential


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(min=1, max=32, multiplier=2),
)
async def acompletion_safe(
    model: str,
    messages: List,
    fallbacks: Optional[List[str]] = None,
    temperature: float = 0.0,
    num_retries: int = 3,
    **kwargs: Any,
) -> str:
    """
    A safe wrapper around litellm.acompletion that handles exceptions and provides fallback models.
    Uses tenacity for automatic retries on rate limits and connection errors.

    # Arguments
        model: Primary model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        messages: List of message dictionaries with role and content
        fallbacks: List of model names to try if primary model fails
        temperature: Controls randomness (0.0 = deterministic, 1.0 = random)
        num_retries: Number of retry attempts per model
        **kwargs: Additional arguments passed to litellm.acompletion

    # Returns
        str: Generated completion text from the model's response

    # Raises
        HTTPException: With appropriate status codes:
        - 401: Authentication error
        - 413: Context window exceeded
        - 400: Bad request or invalid request
        - 503: Service unavailable or all models exhausted
        - 500: OpenAI error or unexpected error

    # Example
    ```python
    response = await acompletion_safe(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        fallbacks=["gpt-3.5-turbo"],
        temperature=0.7
    )
    print(response)  # Generated text from the model
    ```
    """
    models_to_try = [model] + (fallbacks or [])
    last_exception = None

    for current_model in models_to_try:
        try:
            response = await acompletion(
                model=current_model,
                messages=messages,
                temperature=temperature,
                num_retries=num_retries,
                **kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            last_exception = e
            if current_model != models_to_try[-1]:
                continue

            if isinstance(e, AuthenticationError):
                raise HTTPException(status_code=401, detail=str(e))
            elif isinstance(e, ContextWindowExceededError):
                raise HTTPException(status_code=413, detail=str(e))
            elif isinstance(e, (BadRequestError, InvalidRequestError)):
                raise HTTPException(status_code=400, detail=str(e))
            elif isinstance(e, ServiceUnavailableError):
                raise HTTPException(status_code=503, detail=str(e))
            elif isinstance(e, (RateLimitError, APIConnectionError)):
                raise e  # Will be retried by tenacity
            elif isinstance(e, OpenAIError):
                raise HTTPException(status_code=500, detail=str(e))
            else:
                raise HTTPException(
                    status_code=500, detail=f"Unexpected error: {str(e)}"
                )

    raise HTTPException(
        status_code=503,
        detail=f"All models exhausted. Last error: {str(last_exception)}",
    )


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(min=1, max=32, multiplier=2),
)
async def acompletion_parse(
    model: str,
    messages: list,
    response_model: Type[BaseModel],
    fallbacks: Optional[List[str]] = None,
    temperature: float = 0.0,
    num_retries: int = 3,
    **kwargs: Any,
) -> BaseModel:
    """
    A safe wrapper around litellm.acompletion that handles exceptions, provides fallback models,
    and parses the response into a specified Pydantic model using OpenAI's native JSON mode.
    Uses tenacity for automatic retries on rate limits and connection errors.

    # Arguments
        model: Primary model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        messages: List of message dictionaries with role and content
        response_model: Pydantic model class to parse the response into
        fallbacks: List of model names to try if primary model fails
        temperature: Controls randomness (0.0 = deterministic, 1.0 = random)
        num_retries: Number of retry attempts per model
        **kwargs: Additional arguments passed to litellm.acompletion

    # Returns
        BaseModel: Instance of the specified response_model containing parsed response

    # Raises
        HTTPException: With appropriate status codes:
        - 401: Authentication error
        - 413: Context window exceeded
        - 400: Bad request or invalid request
        - 503: Service unavailable or all models exhausted
        - 500: OpenAI error or unexpected error

    # Example
    ```python
    from pydantic import BaseModel

    class ResponseSchema(BaseModel):
        answer: str
        confidence: float

    response = await acompletion_parse(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        response_model=ResponseSchema,
        temperature=0.7
    )
    print(response.answer)      # The answer text
    print(response.confidence)  # Confidence score
    ```
    """
    models_to_try = [model] + (fallbacks or [])
    last_exception = None

    for current_model in models_to_try:
        try:
            response: BaseModel = await acompletion(
                model=current_model,
                messages=messages,
                temperature=temperature,
                response_format=response_model,
                num_retries=num_retries,
                **kwargs,
            )
            return response

        except Exception as e:
            last_exception = e
            if current_model != models_to_try[-1]:
                continue

            if isinstance(e, AuthenticationError):
                raise HTTPException(status_code=401, detail=str(e))
            elif isinstance(e, ContextWindowExceededError):
                raise HTTPException(status_code=413, detail=str(e))
            elif isinstance(e, (BadRequestError, InvalidRequestError)):
                raise HTTPException(status_code=400, detail=str(e))
            elif isinstance(e, ServiceUnavailableError):
                raise HTTPException(status_code=503, detail=str(e))
            elif isinstance(e, (RateLimitError, APIConnectionError)):
                raise e  # Will be retried by tenacity
            elif isinstance(e, OpenAIError):
                raise HTTPException(status_code=500, detail=str(e))
            else:
                raise HTTPException(
                    status_code=500, detail=f"Unexpected error: {str(e)}"
                )

    raise HTTPException(
        status_code=503,
        detail=f"All models exhausted. Last error: {str(last_exception)}",
    )
