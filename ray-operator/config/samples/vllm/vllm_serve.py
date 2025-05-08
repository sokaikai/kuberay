import json
import logging
import os
import time
from typing import AsyncGenerator

from fastapi import BackgroundTasks
from ray import serve
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response, JSONResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
# from vllm.utils import iterate_with_cancellation

# Environment and configuration setup
logger = logging.getLogger("ray.serve")




@serve.deployment(name="VLLMDeployment")
class VLLMDeployment:
    def __init__(self, **kwargs):

        logger.info("Started Loading the model")
        args = AsyncEngineArgs(
            model=str(os.getenv("MODEL_LOCAL_PATH", "Final Model")),
            # Model identifier from Hugging Face Hub or local path.
            dtype=str(os.getenv("MODEL_DTYPE", "auto")),
            # Automatically determine the data type (e.g., float16 or float32) for model weights and computations.
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
            # Percentage of GPU memory to utilize, reserving some for overhead.
            max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
            # Maximum sequence length (in tokens) the model can handle, including both input and output tokens.
            max_num_seqs=int(os.getenv("MAX_NUM_SEQ", "512")),
            # Maximum number of sequences (requests) to process in parallel.
            max_num_batched_tokens=int(os.getenv("MAX_NUM_BATCHED_TOKENS", "32768")),
            # Maximum number of tokens processed in a single batch across all sequences (max_model_len * max_num_seqs).
            trust_remote_code=True,  # Allow execution of untrusted code from the model repository (use with caution).
            enable_chunked_prefill=False,  # Disable chunked prefill to avoid compatibility issues with prefix caching.
            max_parallel_loading_workers=int(os.getenv("PARALLEL_LOADING_WORKERS", 2)),
            # Number of parallel workers to load the model concurrently.
            pipeline_parallel_size=int(os.getenv("PIPELINE_PARALLELISM", 1)),
            # Number of pipeline parallelism stages; typically set to 1 unless using model parallelism.
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLELISM", 1)),
            # Number of tensor parallelism stages; typically set to 1 unless using model parallelism.
            enable_prefix_caching=True,  # Enable prefix caching to improve performance for similar prompt prefixes.
            quantization=os.getenv("QUANTIZATION", None),  # Model Quantization
            enforce_eager=True,
            disable_log_requests=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.max_model_len = args.max_model_len
        logger.info("Loaded the VLLM Model")
        logger.info(f"VLLM Engine initialized with max_model_len: {self.max_model_len}")

    async def stream_results(self, results_generator) -> AsyncGenerator[bytes, None]:
        num_returned = 0
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            ret = {"text": text_output}
            yield (json.dumps(ret) + "\n").encode("utf-8")
            num_returned += len(text_output)

    async def may_abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    async def __call__(self, request: Request) -> Response:
        try:
            request_dict = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON in request body"})

        context_length = request_dict.pop("context_length", 8192)  # Default to 8k

        # Ensure context length is either 8k or 32k
        if context_length not in [8192, 32768]:
            context_length = 8192  # Default to 8k if invalid
        text = request_dict.pop("text")
        stream = request_dict.pop("stream", False)

        default_sampling_params = {
            "temperature": 0,
            "max_tokens": 256,
            "stop": ['}'],
            'include_stop_str_in_output': True
        }

        # Get model config and tokenizer
        model_config = await self.engine.get_model_config()
        tokenizer = await self.engine.get_tokenizer()

        # input_token_ids = tokenizer.encode(prompt)
        # input_tokens = len(input_token_ids)
        # max_possible_new_tokens = min(context_length, model_config.max_model_len) - input_tokens
        # max_new_tokens = min(request_dict.get("max_tokens", 8192), max_possible_new_tokens)

        sampling_params = {**default_sampling_params, **request_dict}
        sampling_params = SamplingParams(**sampling_params)

        request_id = random_uuid()
        start_time = time.time()
        logger.info('Started processing request with id: {} and text: {}'.format(request_id, text))


        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        # results_generator = iterate_with_cancellation(
        #     results_generator, is_cancelled=request.is_disconnected)

        if stream:
            background_tasks = BackgroundTasks()
            # Using background_tasks to abort the request
            # if the client disconnects.
            background_tasks.add_task(self.may_abort_request, request_id)
            return StreamingResponse(
                self.stream_results(results_generator), background=background_tasks
            )

        # Non-streaming case
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                logger.warning(f"Client disconnected for request {request_id}")
                return Response(status_code=499)
            final_output = request_output

        assert final_output is not None
        ret = {"results": json.loads(final_output)}
        ret = {"results": json.loads(final_output)}
        logger.info('Completed processing request with id: {} in {} secs'.format(
            request_id,
            time.time() - start_time
        ))

        return Response(content=json.dumps(ret))


model = VLLMDeployment.bind()