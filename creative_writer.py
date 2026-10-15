#!/usr/bin/env python3
"""
Creative Writer CLI - generates several versions of a creative text with the OpenAI API.

Setup:
    pip install openai
    export OPENAI_API_KEY="sk-..."          (Windows PowerShell: $env:OPENAI_API_KEY="sk-...")

Examples:
    python creative_writer.py "Slogan for a reusable coffee cup brand" --type slogan
    python creative_writer.py "Blog post about learning Git" --type seo --keywords "git tutorial" --words 300
    python creative_writer.py "Poem about autumn in Finland" --type poem --temperature 1.2
    python creative_writer.py --type marketing          (asks for the request interactively)

Every version is generated with its own API request (no `n` parameter), so the
results can be compared for creativity, repetition and latency.
"""

import argparse
import os
import sys
import time

try:
    from openai import (
        OpenAI,
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        NotFoundError,
        RateLimitError,
    )
except ImportError:
    sys.exit("The 'openai' package is not installed. Run: pip install openai")


DEFAULT_MODEL = "gpt-4o-mini"

BASE_SYSTEM_PROMPT = """You are a professional creative writer and copywriter.

Rules:
- Write original work only. Never copy or imitate the actual text of existing songs, poems, slogans or articles.
- Follow the requested format, tone and audience exactly.
- Do not invent facts, statistics, quotes, prices or product features. If a detail is missing, write around it or use a clearly marked placeholder such as [product name].
- Output only the finished text: no introductions, no explanations, no notes about what you did.
- Keep within the requested length."""

TYPE_GUIDANCE = {
    "general": "Write the requested creative text.",
    "marketing": "Write persuasive marketing copy: a strong hook, clear benefits for the reader and a call to action.",
    "slogan": "Write one short, memorable slogan of at most 10 words. Output only the slogan.",
    "meme": "Write meme text: a top line and a bottom line, short and punchy. Format: TOP: ... / BOTTOM: ...",
    "poem": "Write an original poem with vivid imagery. Rhyme only if it sounds natural.",
    "lyrics": "Write original song lyrics with labelled sections (Verse, Chorus, Bridge) and a catchy, repeatable chorus.",
    "blog": "Write a blog post with a title, a short engaging introduction, a few sections with subheadings and a conclusion.",
    "seo": (
        "Write an SEO-oriented article with a title, subheadings and a short meta description at the end.\n"
        "Use the main keyword naturally in the title, the first paragraph and at most a few more times.\n"
        "Instead of repeating it, use synonyms and semantically related terms that a reader of this topic would expect. "
        "Never stuff keywords; the text must read naturally for a human."
    ),
}


def build_system_prompt(text_type: str, words: int | None, keywords: str | None) -> str:
    parts = [BASE_SYSTEM_PROMPT, "", "Task type: " + TYPE_GUIDANCE[text_type]]
    if words:
        parts.append(f"Target length: about {words} words.")
    if keywords:
        parts.append(f"Main keyword(s): {keywords}")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate several versions of a creative text with the OpenAI API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("request", nargs="?", help="What to write. If omitted, you will be asked for it.")
    p.add_argument("--type", choices=sorted(TYPE_GUIDANCE), default="general", help="Kind of text")
    p.add_argument("--keywords", help="Main keyword(s), mainly for --type seo")
    p.add_argument("--versions", type=int, default=3, help="Number of versions (one request each)")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), help="Model name")

    length = p.add_argument_group("length limits")
    length.add_argument("--words", type=int, help="Soft limit: target word count told to the model")
    length.add_argument("--max-tokens", type=int, default=600, help="Hard limit on generated tokens")

    ctl = p.add_argument_group("generation controls (only sent if set; support depends on the model)")
    ctl.add_argument("--temperature", type=float, help="Randomness, usually 0-2")
    ctl.add_argument("--top-p", type=float, help="Nucleus sampling, 0-1")
    ctl.add_argument("--presence-penalty", type=float, help="-2 to 2, encourages new topics")
    ctl.add_argument("--frequency-penalty", type=float, help="-2 to 2, discourages repeating words")
    ctl.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"],
                     help="Only for reasoning models")
    return p.parse_args()


def validate(args: argparse.Namespace) -> None:
    checks = [
        (args.versions < 1, "--versions must be at least 1"),
        (args.max_tokens < 1, "--max-tokens must be at least 1"),
        (args.words is not None and args.words < 1, "--words must be at least 1"),
        (args.temperature is not None and not 0 <= args.temperature <= 2, "--temperature must be between 0 and 2"),
        (args.top_p is not None and not 0 <= args.top_p <= 1, "--top-p must be between 0 and 1"),
        (args.presence_penalty is not None and not -2 <= args.presence_penalty <= 2,
         "--presence-penalty must be between -2 and 2"),
        (args.frequency_penalty is not None and not -2 <= args.frequency_penalty <= 2,
         "--frequency-penalty must be between -2 and 2"),
    ]
    for failed, message in checks:
        if failed:
            sys.exit("Error: " + message)


def get_request(args: argparse.Namespace) -> str:
    if args.request:
        return args.request.strip()
    try:
        text = input("What should I write? > ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nNo request given.")
    if not text:
        sys.exit("No request given.")
    return text


def build_params(args: argparse.Namespace, system_prompt: str, request: str) -> dict:
    params = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ],
    }
    # OpenAI's newer models want max_completion_tokens; most local / compatible servers use max_tokens.
    if os.getenv("OPENAI_BASE_URL"):
        params["max_tokens"] = args.max_tokens
    else:
        params["max_completion_tokens"] = args.max_tokens

    optional = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "reasoning_effort": args.reasoning_effort,
    }
    params.update({k: v for k, v in optional.items() if v is not None})
    return params


def generate(client: OpenAI, params: dict) -> dict:
    """One API request. Returns a result dict; raises only for errors that should stop the program."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**params)
    except AuthenticationError:
        raise SystemExit("Authentication failed: check that OPENAI_API_KEY is correct and still active.")
    except NotFoundError:
        raise SystemExit(f"Model '{params['model']}' was not found or you have no access to it. Try --model.")
    except BadRequestError as e:
        # Typical cause: a control the model does not support, e.g. temperature on a reasoning model.
        raise SystemExit(f"The API rejected the request: {error_text(e)}\n"
                         "Tip: some models do not support every control. Remove the option it mentions.")
    except RateLimitError as e:
        return {"error": f"Rate limit or quota exceeded: {error_text(e)}"}
    except APITimeoutError:
        return {"error": "The request timed out."}
    except APIConnectionError:
        return {"error": "Could not connect to the API. Check your internet connection."}
    except APIStatusError as e:
        return {"error": f"API error {e.status_code}: {error_text(e)}"}

    elapsed = time.perf_counter() - start
    choice = response.choices[0]
    usage = response.usage
    return {
        "text": (choice.message.content or "").strip(),
        "finish_reason": choice.finish_reason,
        "seconds": elapsed,
        "output_tokens": usage.completion_tokens if usage else None,
        "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
        if usage else None,
    }


def error_text(e: Exception) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        message = body.get("message") or (body.get("error") or {}).get("message")
        if message:
            return message
    return str(e)


def print_result(index: int, total: int, result: dict) -> None:
    print(f"\n{'=' * 20} Version {index}/{total} {'=' * 20}")
    if "error" in result:
        print(f"[Failed] {result['error']}")
        return
    print(result["text"] or "[The model returned no text.]")
    stats = [f"{result['seconds']:.1f} s"]
    if result["output_tokens"] is not None:
        stats.append(f"{result['output_tokens']} output tokens")
    if result["reasoning_tokens"]:
        stats.append(f"{result['reasoning_tokens']} of them reasoning")
    print(f"\n({', '.join(stats)})")
    if result["finish_reason"] == "length":
        print("[Note] The text was cut off by --max-tokens. Raise it for a complete text.")


def main() -> None:
    args = parse_args()
    validate(args)

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Set it as an environment variable, e.g.\n"
                 '  macOS/Linux:  export OPENAI_API_KEY="sk-..."\n'
                 '  PowerShell:   $env:OPENAI_API_KEY="sk-..."')

    request = get_request(args)
    system_prompt = build_system_prompt(args.type, args.words, args.keywords)
    params = build_params(args, system_prompt, request)
    client = OpenAI(timeout=60, max_retries=2)  # reads OPENAI_API_KEY (and OPENAI_BASE_URL) itself

    settings = {k: v for k, v in params.items() if k not in ("messages",)}
    print(f"Request: {request}")
    print(f"Type: {args.type} | Settings: {settings}")

    results = []
    for i in range(1, args.versions + 1):
        result = generate(client, params)
        results.append(result)
        print_result(i, args.versions, result)

    done = [r for r in results if "error" not in r]
    print(f"\n{'=' * 52}")
    if done:
        avg = sum(r["seconds"] for r in done) / len(done)
        print(f"{len(done)}/{len(results)} versions generated, average latency {avg:.1f} s.")
    else:
        print("No versions could be generated.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
