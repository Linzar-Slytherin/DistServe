import argparse
from distserve import OfflineLLM, SamplingParams
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='The model to use', default='facebook/opt-6.7b')
args = parser.parse_args()

# Sample prompts.
prompts = [
    "The sun sets behind the hills. Painting the sky in hues of orange. Fading",
    "A gentle breeze whispers. Through the leaves. As the sun dips",
    "In the stillness of night. Stars begin to appear. Watching",
    "Courage is not the absence of fear. It is the strength. To move forward",
    "Raindrops dance on rooftops. Creating a rhythm. That echoes through",
    "The mountain stood tall. A silent witness. To the passage of time",
    "Dreams are the seeds. Planted deep within the soul. Waiting to bloom",
    "She walked along the beach. Feeling the cool sand. Beneath her feet",
    "The river flowed endlessly. Carving a path. Through the earth",
    "Under the shade of an ancient tree. He found a moment. Of peace",
    "The city never sleeps. Its lights twinkling. Like distant stars",
    "Hope is a fragile thing. Yet powerful enough. To ignite change",
    "Waves crashed against rocks. Their white foam. Disappearing swiftly",
    "The clock ticked slowly. Marking each second. Passing unnoticed",
    "In the heart of the forest. A small creature. Watched with curiosity",
    "The moon hung low. Casting a silver glow. On the quiet town",
    "He climbed the stairs. One step at a time. Breathing heavily",
    "A song of joy rose. From the valley below. Echoing far",
    "Time moves forward. Leaving behind only. Shadows of yesterday",
    "A child’s laughter filled the air. Reminding us all. Of simple joys",
    "In the silence of the library. She could hear her heart. Beating steadily",
    "The old oak tree. With branches spread wide. Stood unmoved",
    "The train whistled sharply. As it sped away. Into the distance",
    "The morning dew sparkled. On the grass. Under the rising sun",
    "She picked up her pen. Ready to write. The story within her",
    "With each passing day. Flowers in the garden. Bloomed more vibrantly",
    "A rainbow appeared after rain. Bringing hope. To all who saw",
    "The candle flickered. Casting shadows that danced. On the walls",
    "The desert stretched endlessly. Golden sands. Glowing under the sun",
    "He whispered softly. Words barely audible. Over the sound of wind",
    "The fire crackled gently. Warming the cold room. Filling it with light",
    "A gentle snow began to fall. Covering everything. In a blanket of white",
    "The horizon glowed bright. As the sun. Rose to greet the day",
    "Leaves crunched beneath. His steady steps. Walking through autumn woods",
    "The violin played softly. A mournful tune. That touched every heart",
    "She opened the door slowly. Unsure of what. Lay waiting beyond",
    "The sky was a canvas. Painted in shades. Of pink and lavender",
    "Stars dotted the sky. Forming patterns. Only dreamers could see",
    "The cat watched intently. As the mouse. Moved closer to danger",
    "He took a deep breath. Ready to face. Whatever lay ahead"
]


# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95, max_tokens=64, stop=["\n"]
)

# Create an LLM for offline inference.
llm = OfflineLLM(
    model_config=ModelConfig(
        model=args.model,
        tokenizer=None
    ),
    disagg_parallel_config=DisaggParallelConfig(
        context=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        ),
        decoding=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
    ),
    cache_config=CacheConfig(
        block_size=16,
        max_num_blocks_per_req=1024,
        gpu_memory_utilization=0.9,
        cpu_swap_space=1.0
    ),
    context_sched_config=ContextStageSchedConfig(
        policy="fcfs",
        max_batch_size=4,
        max_tokens_per_batch=16384
    ),
    decoding_sched_config=DecodingStageSchedConfig(
        policy="fcfs",
        max_batch_size=4,
        max_tokens_per_batch=16384
    )
)

# Generate texts from the prompts. The output is a list of Request objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

# Print the outputs.
for prompt, step_outputs in zip(prompts, outputs):
    # new_token_ids = [step_output.new_token_id for step_output in step_outputs]
    # output_text = llm.tokenizer.decode(new_token_ids)
    print(
        f"Prompt: {prompt!r}, Generated text: {' '.join([step_output.new_token for step_output in step_outputs])} ({len(step_outputs)} tokens generated)."
    )
