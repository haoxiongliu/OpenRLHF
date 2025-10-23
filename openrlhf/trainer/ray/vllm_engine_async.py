import asyncio
import os
import yaml

import ray
import torch
import random

from .vllm_engine import BaseLLMRayActor


@ray.remote
class AgentInstance:
    def __init__(self, agent_func_path):
        if agent_func_path.endswith(".py"):
            import importlib.util

            spec = importlib.util.spec_from_file_location("step", agent_func_path)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)
            self.agent_step = agent_module.step
        else:
            raise ValueError("Agent path must be a Python file")

    async def step(self, observation, action, label, **kwargs):
        return await self.agent_step(observation, action, label, **kwargs)


@ray.remote
def get_tokenize_text_len(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0])


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        self.agent_func_path = kwargs.pop("agent_func_path")
        self.proofaug_config_path = kwargs.pop("proofaug_config_path", "configs/default.yaml")
        print(f"proofaug_config_path: {self.proofaug_config_path}")
        # Initialize super class
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        # Initialize result queue for streaming completed results
        self.result_queue = asyncio.Queue()

        os.environ["VLLM_USE_V1"] = "1"
        import vllm

        assert vllm.__version__ > "0.8.5", "Asyn VLLM version must be greater than 0.8.5"

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    async def add_requests(self, 
            sampling_params, prompts, labels, 
            max_length, hf_tokenizer=None, max_steps=10000
        ):
        """
        Process requests from rank0 and generate responses with multiple agent interactions.
        Each prompt will go through multiple steps of interaction using the step function.
        Results are streamed back as each agent completes its execution.

        Args:
            sampling_params: Parameters for sampling
            prompts: List of prompts to process
            labels: List of labels corresponding to prompts
            max_steps: Maximum number of interaction steps
        """

        # Create semaphore to control concurrent task execution
        NUM_TASKS = int(os.environ.get("OPENRLHF_ASYNC_NUM_TASKS", 128))
        semaphore = asyncio.Semaphore(NUM_TASKS)
        with open(self.proofaug_config_path, 'r') as f:
            proofaug_config : dict = yaml.safe_load(f)
        
        # Check if conservative mode is enabled
        conservative_mode = proofaug_config.get("conservative", False)
        proofaug = proofaug_config.get("proofaug", False)
        
        if conservative_mode and proofaug:
            await self._add_requests_conservative(sampling_params, prompts, labels, max_length, hf_tokenizer, max_steps, proofaug_config, semaphore)
        else:
            await self._add_requests_original(sampling_params, prompts, labels, max_length, hf_tokenizer, max_steps, proofaug_config, semaphore)

    async def _add_requests_original(self, sampling_params, prompts, labels, max_length, hf_tokenizer, max_steps, proofaug_config, semaphore):
        """Original logic for processing requests"""
        async def execute_agent(prompt, label, sampling_params):
            async with semaphore:
                # Create a unique agent instance for this prompt
                agent_instance = AgentInstance.remote(self.agent_func_path)

                # Initialize observations and actions for the current prompt
                observation = prompt
                action_ranges = []
                total_reward = 0
                total_orig_reward = 0
                final_scores = 0
                extra_logs = None

                # Execute multiple steps of interaction
                sample_original_action = None
                sample_structured_output = None
                for step_idx in range(max_steps):
                    # Next sampling budget
                    observation_tokens_len = len(
                        hf_tokenizer(observation, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                    )
                    sampling_params.max_tokens = max_length - observation_tokens_len
                    # No budget to generate, break
                    if sampling_params.max_tokens <= 0:
                        break

                    # Generate response asynchronously
                    request_output = await self.generate_async(observation, sampling_params)
                    action = request_output.outputs[0].text
                    original_action_len = len(action)
                    action_start = len(observation)

                    # Call step function to get reward and next observation
                    # Use asyncio.to_thread to make Ray remote call non-blocking
                    # Load kwargs config from YAML file
                    kwargs = {"sampling_params": sampling_params,
                              "proofaug_config": proofaug_config}
                    result = await agent_instance.step.remote(observation, action, label, **kwargs)
                    reward = result["rewards"]
                    orig_reward = result["orig_rewards"]
                    if isinstance(reward, torch.Tensor):
                        reward = reward.item()
                    if isinstance(orig_reward, torch.Tensor):
                        orig_reward = orig_reward.item()
                    total_reward += reward
                    total_orig_reward += orig_reward
                    final_scores = result.get("scores", total_reward)
                    observation = result["next_observation"]
                    done = result["done"]
                    extra_logs = result.get("extra_logs", {})

                    orig_reward = extra_logs.get("orig_rewards", 0.0) # require float
                    if isinstance(orig_reward, torch.Tensor):
                        orig_reward = orig_reward.item()

                    # consider structured output from the environment
                    action_end = len(observation)
                    action_ranges.append((action_start, action_end))
                    if original_action_len != action_end - action_start and random.random() < 0.1:
                        sample_original_action = action
                        sample_structured_output = observation[action_start:action_end]

                    # Get sampling params from the environment step
                    if result.get("sampling_params", None):
                        sampling_params = result["sampling_params"]

                    if done:
                        break
                if sample_structured_output is not None:
                    print(f"structured output detected:\n\n{sample_original_action}\n\ntransformed to\n\n{sample_structured_output}")
                ray.kill(agent_instance)

                # Store the final response when agent execution is complete
                # this is used in experience_maker_async.py, line 40.
                final_response = {
                    "prompt": prompt,
                    "label": label,
                    "observation": observation,
                    "reward": total_reward,
                    "orig_reward": total_orig_reward,
                    "scores": final_scores,
                    "extra_logs": extra_logs,
                    "action_ranges": action_ranges,
                }
                await self.result_queue.put(final_response)

        # Create and start tasks for all agent executions with controlled concurrency
        import copy

        tasks = []
        for prompt, label in zip(prompts, labels):
            tasks.append(execute_agent(prompt, label, copy.deepcopy(sampling_params)))

        # Run the async code using the class's event loop
        await asyncio.gather(*tasks)

    async def _add_requests_conservative(self, sampling_params, prompts, labels, max_length, hf_tokenizer, max_steps, proofaug_config: dict, semaphore):
        """Conservative logic: only use proofaug replacement when all original responses for the same prompt have reward=0"""
        import copy
        
        # Group consecutive identical prompts
        prompt_groups = []
        current_group = []
        current_prompt = None
        
        for i, (prompt, label) in enumerate(zip(prompts, labels)):
            if prompt != current_prompt:
                if current_group:
                    prompt_groups.append(current_group)
                current_group = [(i, prompt, label)]
                current_prompt = prompt
            else:
                current_group.append((i, prompt, label))
        
        if current_group:
            prompt_groups.append(current_group)
        
        # Store results with original indices to maintain order
        final_results = [None] * len(prompts)
        
        async def execute_first_pass(prompt, label):
            async with semaphore:
                # Create a unique agent instance for this prompt
                agent_instance = AgentInstance.remote(self.agent_func_path)

                # Initialize observations and actions for the current prompt
                observation = prompt
                action_ranges = []
                total_reward = 0
                total_orig_reward = 0
                final_scores = 0
                extra_logs = None
                
                # Store all sampling steps for potential reuse
                sampling_steps = []

                # Execute multiple steps of interaction with sampling
                for step_idx in range(max_steps):
                    # Next sampling budget
                    observation_tokens_len = len(
                        hf_tokenizer(observation, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                    )
                    current_sampling_params = copy.deepcopy(sampling_params)
                    current_sampling_params.max_tokens = max_length - observation_tokens_len
                    # No budget to generate, break
                    if current_sampling_params.max_tokens <= 0:
                        break

                    # Generate response asynchronously
                    request_output = await self.generate_async(observation, current_sampling_params)
                    action = request_output.outputs[0].text
                    original_action_len = len(action)
                    action_start = len(observation)
                    
                    # Store the sampling step
                    sampling_steps.append({
                        'observation_before': observation,
                        'action': action,
                        'action_start': action_start,
                        'original_action_len': original_action_len
                    })

                    # Call step function with proofaug disabled for first evaluation
                    first_pass_config = copy.deepcopy(proofaug_config)
                    first_pass_config["record_pa_reward"] = True
                    first_pass_config["proofaug"] = False
                    
                    kwargs = {"sampling_params": current_sampling_params,
                              "proofaug_config": first_pass_config}
                    result = await agent_instance.step.remote(observation, action, label, **kwargs)
                    reward = result["rewards"]
                    orig_reward = result["orig_rewards"]
                    if isinstance(reward, torch.Tensor):
                        reward = reward.item()
                    if isinstance(orig_reward, torch.Tensor):
                        orig_reward = orig_reward.item()
                    total_reward += reward
                    total_orig_reward += orig_reward
                    final_scores = result.get("scores", total_reward)
                    observation = result["next_observation"]
                    done = result["done"]
                    extra_logs = result.get("extra_logs", {})

                    orig_reward = extra_logs.get("orig_rewards", 0.0) # require float
                    if isinstance(orig_reward, torch.Tensor):
                        orig_reward = orig_reward.item()

                    # consider structured output from the environment
                    action_end = len(observation)
                    action_ranges.append((action_start, action_end))
                    assert original_action_len == action_end - action_start, f"in conservative 1st pass, there must be no change in action length"

                    # Get sampling params from the environment step
                    if result.get("sampling_params", None):
                        current_sampling_params = result["sampling_params"]

                    if done:
                        break
                        
                ray.kill(agent_instance)

                # Return both the result and sampling data
                first_pass_result = {
                    "prompt": prompt,
                    "label": label,
                    "observation": observation,
                    "reward": total_reward,
                    "orig_reward": total_orig_reward,
                    "scores": final_scores,
                    "extra_logs": extra_logs,
                    "action_ranges": action_ranges,
                }
                return first_pass_result, sampling_steps

        async def execute_second_pass(prompt, label, sampling_steps):
            async with semaphore:
                # Create a new agent instance for re-evaluation
                agent_instance = AgentInstance.remote(self.agent_func_path)

                # Replay the same sampling steps but with proofaug enabled
                observation = prompt
                action_ranges = []
                total_reward = 0
                total_orig_reward = 0
                final_scores = 0
                extra_logs = None
                
                sample_original_action = None
                sample_structured_output = None
                
                for step_data in sampling_steps:
                    observation = step_data['observation_before']
                    action = step_data['action']
                    action_start = step_data['action_start']
                    original_action_len = step_data['original_action_len']

                    # Re-evaluate with original proofaug config
                    kwargs = {"sampling_params": sampling_params,
                              "proofaug_config": proofaug_config}
                    result = await agent_instance.step.remote(observation, action, label, **kwargs)
                    reward = result["rewards"]
                    orig_reward = result["orig_rewards"]
                    if isinstance(reward, torch.Tensor):
                        reward = reward.item()
                    if isinstance(orig_reward, torch.Tensor):
                        orig_reward = orig_reward.item()
                    total_reward += reward
                    total_orig_reward += orig_reward
                    final_scores = result.get("scores", total_reward)
                    observation = result["next_observation"]
                    done = result["done"]
                    extra_logs = result.get("extra_logs", {})

                    orig_reward = extra_logs.get("orig_rewards", 0.0) # require float
                    if isinstance(orig_reward, torch.Tensor):
                        orig_reward = orig_reward.item()

                    # consider structured output from the environment
                    action_end = len(observation)
                    action_ranges.append((action_start, action_end))
                    if original_action_len != action_end - action_start and random.random() < 0.1:
                        sample_original_action = action
                        sample_structured_output = observation[action_start:action_end]

                    if done:
                        break
                        
                if sample_structured_output is not None:
                    print(f"structured output detected:\n\n{sample_original_action}\n\ntransformed to\n\n{sample_structured_output}")
                ray.kill(agent_instance)

                # Return the second pass result
                second_pass_result = {
                    "prompt": prompt,
                    "label": label,
                    "observation": observation,
                    "reward": total_reward,
                    "orig_reward": total_orig_reward,
                    "scores": final_scores,
                    "extra_logs": extra_logs,
                    "action_ranges": action_ranges,
                }
                return second_pass_result
        
        # Process each group
        for group in prompt_groups:
            # First pass: concurrent sampling and evaluation with proofaug disabled
            first_pass_tasks = []
            for original_idx, prompt, label in group:
                first_pass_tasks.append(execute_first_pass(prompt, label))
            
            # Wait for all first pass results - this maintains order
            first_pass_data = await asyncio.gather(*first_pass_tasks)
            first_pass_results = [data[0] for data in first_pass_data]
            group_samples = [data[1] for data in first_pass_data]
            
            # Check if all original responses have reward = 0
            all_zero_rewards = all(result["reward"] == 0.0 for result in first_pass_results)
            
            # If pa_reward is also 0, proofaug won't help, so skip second pass
            all_zero_pa_rewards = True
            for result in first_pass_results:
                pa_reward = result["extra_logs"]["pa_rewards"]
                if pa_reward != 0.0:
                    all_zero_pa_rewards = False
                    break
            
            # Only do second pass if rewards are 0 but pa_rewards indicate potential improvement
            if all_zero_rewards and not all_zero_pa_rewards:
                # Second pass: concurrent re-evaluation with proofaug enabled
                second_pass_tasks = []
                for i, (original_idx, prompt, label) in enumerate(group):
                    second_pass_tasks.append(execute_second_pass(prompt, label, group_samples[i]))
                
                second_pass_results = await asyncio.gather(*second_pass_tasks)
                
                # Use second pass results
                for i, (original_idx, _, _) in enumerate(group):
                    final_results[original_idx] = second_pass_results[i]
            else:
                # Use first pass results
                for i, (original_idx, _, _) in enumerate(group):
                    final_results[original_idx] = first_pass_results[i]
        
        # Put all results into the queue in original order
        for result in final_results:
            await self.result_queue.put(result)

    async def generate_async(self, prompts, sampling_params):
        from vllm.utils import random_uuid

        request_id = random_uuid()
        results_generator = self.llm.generate(prompts, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output

    async def get_responses(self):
        """
        Synchronously get all completed agent results from the queue.
        Waits for all tasks to complete before returning results.
        Returns: List of all completed agent results.
        """
        # Get all results from the queue
        results = []
        while not self.result_queue.empty():
            try:
                results.append(await self.result_queue.get())
            except asyncio.QueueEmpty:
                break
        return results
