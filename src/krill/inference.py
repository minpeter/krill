"""
Module for interactive inference using streaming text generation.
"""
import transformers
import torch
import sys
from threading import Thread
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoTokenizer, GenerationConfig


def do_inference(model_id: str, inspect: bool = False):
    """Interactive inference on a text generation model with streaming output."""
    print(f"⚓️ Loading model: {model_id}...")

    if inspect:
        print("🔍 Inspect mode enabled (experimental)")

    # Load tokenizer and model pipeline
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # pick GPU if available, else CPU
        device = 0 if torch.cuda.is_available() else -1

        # load model and move it to the selected device
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            use_cache=True,
            attn_implementation="eager"
        ).to("cuda" if device == 0 else "cpu")

        # set up pipeline on same device
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch.bfloat16
        )
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("✅ Model loaded! Enter 'quit' or 'exit' to quit. 🏴‍☠️")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    generation_config = GenerationConfig(
        max_new_tokens=128,
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

            generated_text = ""
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
                generated_text += new_text

            print()

            if inspect:
                from krill.utils.inspect_model import inspect_token_predictions

                full_text = prompt + generated_text
                inspect_token_predictions(tokenizer, model, full_text)
            thread.join()

        except KeyboardInterrupt:
            print("\nSee you next time! ⛵️")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")
