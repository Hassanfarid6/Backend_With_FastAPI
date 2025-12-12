# Gemini Model Integration

This document outlines the integration and usage of Gemini models within this project.

## `GeminiModel` Class

This class provides a simplified interface for interacting with the Gemini API, featuring retry mechanisms and parallel processing for multiple prompts.

```python
class GeminiModel:
    """Class for the Gemini model."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-001",
        finetuned_model: bool = False,
        distribute_requests: bool = False,
        cache_name: str | None = None,
        temperature: float = 0.01,
        **kwargs,
    ):
        self.model_name = model_name
        self.finetuned_model = finetuned_model
        self.arguments = kwargs
        self.distribute_requests = distribute_requests
        self.temperature = temperature
        model_name = self.model_name
        if not self.finetuned_model and self.distribute_requests:
            random_region = random.choice(GEMINI_AVAILABLE_REGIONS)
            model_name = GEMINI_URL.format(
                GCP_PROJECT=GCP_PROJECT,
                region=random_region,
                model_name=self.model_name,
            )
        if cache_name is not None:
            cached_content = caching.CachedContent(cached_content_name=cache_name)
            self.model = GenerativeModel.from_cached_content(
                cached_content=cached_content
            )
        else:
            self.model = GenerativeModel(model_name=model_name)

    @retry(max_attempts=12, base_delay=2, backoff_factor=2)
    def call(self, prompt: str, parser_func=None) -> str:
        """Calls the Gemini model with the given prompt.

        Args:
            prompt (str): The prompt to call the model with.
            parser_func (callable, optional): A function that processes the LLM
              output. It takes the model"s response as input and returns the
              processed result.

        Returns:
            str: The processed response from the model.
        """
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=self.temperature,
                **self.arguments,
            ),
            safety_settings=SAFETY_FILTER_CONFIG,
        ).text
        if parser_func:
            return parser_func(response)
        return response

    def call_parallel(
        self,
        prompts: List[str],
        parser_func: Optional[Callable[[str], str]] = None,
        timeout: int = 60,
        max_retries: int = 5,
    ) -> List[Optional[str]]:
        """Calls the Gemini model for multiple prompts in parallel using threads with retry logic.

        Args:
            prompts (List[str]): A list of prompts to call the model with.
            parser_func (callable, optional): A function to process each response.
            timeout (int): The maximum time (in seconds) to wait for each thread.
            max_retries (int): The maximum number of retries for timed-out threads.

        Returns:
            List[Optional[str]]:
            A list of responses, or None for threads that failed.
        """
        results = [None] * len(prompts)

        def worker(index: int, prompt: str):
            """Thread worker function to call the model and store the result with retries."""
            retries = 0
            while retries <= max_retries:
                try:
                    return self.call(prompt, parser_func)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Error for prompt {index}: {str(e)}")
                    retries += 1
                    if retries <= max_retries:
                        print(f"Retrying ({retries}/{max_retries}) for prompt {index}")
                        time.sleep(1)  # Small delay before retrying
                    else:
                        return f"Error after retries: {str(e)}"

        # Create and start one thread for each prompt
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            future_to_index = {
                executor.submit(worker, i, prompt): i
                for i, prompt in enumerate(prompts)
            }

            for future in as_completed(future_to_index, timeout=timeout):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Unhandled error for prompt {index}: {e}")
                    results[index] = "Unhandled Error"

        # Handle remaining unfinished tasks after the timeout
        for future in future_to_index:
            index = future_to_index[future]
            if not future.done():
                print(f"Timeout occurred for prompt {index}")
                results[index] = "Timeout"

        return results
```

## `Gemini` (BaseLlm Integration)

This class extends `BaseLlm` for robust Gemini model integration, supporting asynchronous content generation and handling different API backends like Vertex AI and the Gemini API.

```python
class Gemini(BaseLlm):
  """Integration for Gemini models.

  Attributes:
    model: The name of the Gemini model.
  """

  model: str = 'gemini-1.5-flash'

  @staticmethod
  @override
  def supported_models() -> list[str]:
    """Provides the list of supported models.

    Returns:
      A list of supported models.
    """

    return [
        r'gemini-.*',
        # fine-tuned vertex endpoint pattern
        r'projects\/.+\/locations\/.+\/endpoints\/.+',
        # vertex gemini long name
        r'projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+',
    ]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Sends a request to the Gemini model.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """
    self._preprocess_request(llm_request)
    self._maybe_append_user_content(llm_request)
    logger.info(
        'Sending out request, model: %s, backend: %s, stream: %s',
        llm_request.model,
        self._api_backend,
        stream,
    )
    logger.info(_build_request_log(llm_request))

    # add tracking headers to custom headers given it will override the headers
    # set in the api client constructor
    if llm_request.config and llm_request.config.http_options:
      if not llm_request.config.http_options.headers:
        llm_request.config.http_options.headers = {}
      llm_request.config.http_options.headers.update(self._tracking_headers)

    if stream:
      responses = await self.api_client.aio.models.generate_content_stream(
          model=llm_request.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )
      response = None
      thought_text = ''
      text = ''
      usage_metadata = None
      # for sse, similar as bidi (see receive method in gemini_llm_connecton.py),
      # we need to mark those text content as partial and after all partial
      # contents are sent, we send an accumulated event which contains all the
      # previous partial content. The only difference is bidi rely on
      # complete_turn flag to detect end while sse depends on finish_reason.
      async for response in responses:
        logger.info(_build_response_log(response))
        llm_response = LlmResponse.create(response)
        usage_metadata = llm_response.usage_metadata
        if (
            llm_response.content
            and llm_response.content.parts
            and llm_response.content.parts[0].text
        ):
          part0 = llm_response.content.parts[0]
          if part0.thought:
            thought_text += part0.text
          else:
            text += part0.text
          llm_response.partial = True
        elif (thought_text or text) and (
            not llm_response.content
            or not llm_response.content.parts
            # don't yield the merged text event when receiving audio data
            or not llm_response.content.parts[0].inline_data
        ):
          parts = []
          if thought_text:
            parts.append(types.Part(text=thought_text, thought=True))
          if text:
            parts.append(types.Part.from_text(text=text))
          yield LlmResponse(
              content=types.ModelContent(parts=parts),
              usage_metadata=llm_response.usage_metadata,
          )
          thought_text = ''
          text = ''
        yield llm_response
      if (
          (text or thought_text)
          and response
          and response.candidates
          and response.candidates[0].finish_reason == types.FinishReason.STOP
      ):
        parts = []
        if thought_text:
          parts.append(types.Part(text=thought_text, thought=True))
        if text:
          parts.append(types.Part.from_text(text=text))
        yield LlmResponse(
            content=types.ModelContent(parts=parts),
            usage_metadata=usage_metadata,
        )

    else:
      response = await self.api_client.aio.models.generate_content(
          model=llm_request.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )
      logger.info(_build_response_log(response))
      yield LlmResponse.create(response)

  @cached_property
  def api_client(self) -> Client:
    """Provides the api client.

    Returns:
      The api client.
    """
    return Client(
        http_options=types.HttpOptions(headers=self._tracking_headers)
    )

  @cached_property
  def _api_backend(self) -> GoogleLLMVariant:
    return (
        GoogleLLMVariant.VERTEX_AI
        if self.api_client.vertexai
        else GoogleLLMVariant.GEMINI_API
    )

  @cached_property
  def _tracking_headers(self) -> dict[str, str]:
    framework_label = f'google-adk/{version.__version__}'
    if os.environ.get(_AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME):
      framework_label = f'{framework_label}+{_AGENT_ENGINE_TELEMETRY_TAG}'
    language_label = 'gl-python/' + sys.version.split()[0]
    version_header_value = f'{framework_label} {language_label}'
    tracking_headers = {
        'x-goog-api-client': version_header_value,
        'user-agent': version_header_value,
    }
    return tracking_headers

  @cached_property
  def _live_api_version(self) -> str:
    if self._api_backend == GoogleLLMVariant.VERTEX_AI:
      # use beta version for vertex api
      return 'v1beta1'
    else:
      # use v1alpha for using API KEY from Google AI Studio
      return 'v1alpha'

  @cached_property
  def _live_api_client(self) -> Client:
    return Client(
        http_options=types.HttpOptions(
            headers=self._tracking_headers, api_version=self._live_api_version
        )
    )

  @contextlib.asynccontextmanager
  async def connect(self, llm_request: LlmRequest) -> BaseLlmConnection:
    """Connects to the Gemini model and returns an llm connection.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.

    Yields:
      BaseLlmConnection, the connection to the Gemini model.
    """
    # add tracking headers to custom headers and set api_version given
    # the customized http options will override the one set in the api client
    # constructor
    if (
        llm_request.live_connect_config
        and llm_request.live_connect_config.http_options
    ):
      if not llm_request.live_connect_config.http_options.headers:
        llm_request.live_connect_config.http_options.headers = {}
      llm_request.live_connect_config.http_options.headers.update(
          self._tracking_headers
      )
      llm_request.live_connect_config.http_options.api_version = (
          self._live_api_version
      )

    llm_request.live_connect_config.system_instruction = types.Content(
        role='system',
        parts=[
            types.Part.from_text(text=llm_request.config.system_instruction)
        ],
    )
    llm_request.live_connect_config.tools = llm_request.config.tools
    async with self._live_api_client.aio.live.connect(
        model=llm_request.model, config=llm_request.live_connect_config
    ) as live_session:
      yield GeminiLlmConnection(live_session)

  def _preprocess_request(self, llm_request: LlmRequest) -> None:

    if self._api_backend == GoogleLLMVariant.GEMINI_API:
      # Using API key from Google AI Studio to call model doesn't support labels.
      if llm_request.config:
        llm_request.config.labels = None

      if llm_request.contents:
        for content in llm_request.contents:
          if not content.parts:
            continue
          for part in content.parts:
            _remove_display_name_if_present(part.inline_data)
            _remove_display_name_if_present(part.file_data)
```

## Gemini LLM Initialization

A utility function for quickly initializing the Gemini model.

```python
def gemini_llm():
  return Gemini(model="gemini-1.5-flash")
```

## Schema Conversion Utilities

Functions for converting between Gemini Schema objects and JSON Schema dictionaries, facilitating data validation and interoperability.

### `gemini_to_json_schema`

Converts a Gemini Schema to a JSON Schema dictionary.

```python
def gemini_to_json_schema(gemini_schema: Schema) -> Dict[str, Any]:
  """Converts a Gemini Schema object into a JSON Schema dictionary.

  Args:
      gemini_schema: An instance of the Gemini Schema class.

  Returns:
      A dictionary representing the equivalent JSON Schema.

  Raises:
      TypeError: If the input is not an instance of the expected Schema class.
      ValueError: If an invalid Gemini Type enum value is encountered.
  """
  if not isinstance(gemini_schema, Schema):
    raise TypeError(
        f"Input must be an instance of Schema, got {type(gemini_schema)}"
    )

  json_schema_dict: Dict[str, Any] = {}

  # Map Type
  gemini_type = getattr(gemini_schema, "type", None)
  if gemini_type and gemini_type != Type.TYPE_UNSPECIFIED:
    json_schema_dict["type"] = gemini_type.lower()
  else:
    json_schema_dict["type"] = "null"

  # Map Nullable
  if getattr(gemini_schema, "nullable", None) == True:
    json_schema_dict["nullable"] = True

  # --- Map direct fields ---
  direct_mappings = {
      "title": "title",
      "description": "description",
      "default": "default",
      "enum": "enum",
      "format": "format",
      "example": "example",
  }
  for gemini_key, json_key in direct_mappings.items():
    value = getattr(gemini_schema, gemini_key, None)
    if value is not None:
      json_schema_dict[json_key] = value

  # String validation
  if gemini_type == Type.STRING:
    str_mappings = {
        "pattern": "pattern",
        "min_length": "minLength",
        "max_length": "maxLength",
    }
    for gemini_key, json_key in str_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Number/Integer validation
  if gemini_type in (Type.NUMBER, Type.INTEGER):
    num_mappings = {
        "minimum": "minimum",
        "maximum": "maximum",
    }
    for gemini_key, json_key in num_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Array validation (Recursive call for items)
  if gemini_type == Type.ARRAY:
    items_schema = getattr(gemini_schema, "items", None)
    if items_schema is not None:
      json_schema_dict["items"] = gemini_to_json_schema(items_schema)

    arr_mappings = {
        "min_items": "minItems",
        "max_items": "maxItems",
    }
    for gemini_key, json_key in arr_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Object validation (Recursive call for properties)
  if gemini_type == Type.OBJECT:
    properties_dict = getattr(gemini_schema, "properties", None)
    if properties_dict is not None:
      json_schema_dict["properties"] = {
          prop_name: gemini_to_json_schema(prop_schema)
          for prop_name, prop_schema in properties_dict.items()
      }

    obj_mappings = {
        "required": "required",
        "min_properties": "minProperties",
        "max_properties": "maxProperties",
        # Note: Ignoring 'property_ordering' as it's not standard JSON Schema
    }
    for gemini_key, json_key in obj_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Map anyOf (Recursive call for subschemas)
  any_of_list = getattr(gemini_schema, "any_of", None)
  if any_of_list is not None:
    json_schema_dict["anyOf"] = [
        gemini_to_json_schema(sub_schema) for sub_schema in any_of_list
    ]

  return json_schema_dict
```

### `_to_gemini_schema`

Converts an OpenAPI schema to a Gemini Schema object.

```python
def _to_gemini_schema(openapi_schema: dict[str, Any]) -> Schema:
  """Converts an OpenAPI schema dictionary to a Gemini Schema object."""
  if openapi_schema is None:
    return None

  if not isinstance(openapi_schema, dict):
    raise TypeError("openapi_schema must be a dictionary")

  openapi_schema = _sanitize_schema_formats_for_gemini(openapi_schema)
  return Schema.from_json_schema(
      json_schema=_ExtendedJSONSchema.model_validate(openapi_schema),
      api_option=get_google_llm_variant(),
  )
```

## Internal API Backend Logic

This property determines the specific Google LLM backend being utilized.

```python
def _api_backend(self) -> GoogleLLMVariant:
    return (
        GoogleLLMVariant.VERTEX_AI
        if self.api_client.vertexai
        else GoogleLLMVariant.GEMINI_API
    )
```