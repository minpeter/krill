"""
Module for interactive inference using streaming text generation.
"""
import transformers
import torch
import sys
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, GenerationConfig


def do_inference(model: str):
    """Interactive inference on a text generation model with streaming output."""
    print(f"⚓️ Loading model: {model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("✅ Model loaded! Enter 'quit' or 'exit' to quit. 🏴‍☠️")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        top_p=0.95,
        temperature=0.4,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id
    )

    while True:
        try:
            prompt = input("> ")
            if prompt.lower() in ["quit", "exit"]:
                print("See you next time! ⛵️")
                break

            # Move cursor up one line and return to start to append model output inline
            sys.stdout.write("\x1b[A\r")
            print(f"> {prompt}", end="", flush=True)

            generation_kwargs = {
                "text_inputs": prompt,
                "generation_config": generation_config,
                "streamer": streamer,
            }
            thread = Thread(target=pipeline, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                sys.stdout.write(new_text)
                sys.stdout.flush()

            print()
            thread.join()

        except KeyboardInterrupt:
            print("\nSee you next time! ⛵️")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")
