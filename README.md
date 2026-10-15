# Creative Writer CLI

A command-line program that uses a large language model as a creative writer. It writes slogans, marketing copy, memes, poems, song lyrics, blog posts and SEO articles, and by default produces **three different versions** of each request so they can be compared.

## Features

- Request from the command line or typed interactively
- Three versions by default, each from a **separate API request** (no `n` parameter)
- System prompt tailored to each text type, including natural keyword use for SEO (synonyms and related terms instead of keyword stuffing)
- Output length limits: a soft word target (`--words`) and a hard token cap (`--max-tokens`)
- API key read only from an environment variable
- Graceful handling of API errors (wrong key, no credits, rate limits, unsupported settings, connection problems)
- Shows latency and token usage for each version

## Requirements

- Python 3.10 or newer
- The OpenAI Python library

```
python -m pip install openai
```

## Setup

Set your API key as an environment variable. It is never stored in the code.

PowerShell (Windows):
```
$env:OPENAI_API_KEY="your-key-here"
```

macOS / Linux:
```
export OPENAI_API_KEY="your-key-here"
```

### Using Gemini instead of OpenAI (optional)

The program also works with any OpenAI-compatible endpoint. For example, to use Google's Gemini with a free Gemini API key:

```
$env:OPENAI_API_KEY="your-gemini-key"
$env:OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
python creative_writer.py "Slogan for a reusable coffee cup brand" --type slogan --model gemini-3.5-flash-lite
```

## Usage

```
python creative_writer.py "Slogan for a reusable coffee cup brand" --type slogan
python creative_writer.py "Poem about autumn in Finland" --type poem --words 80
python creative_writer.py "Article about how beginners can learn Git" --type seo --keywords "learn git" --words 250
python creative_writer.py --type marketing
```

The last example asks for the request interactively.

### Options

| Option | Description |
|---|---|
| `--type` | `general`, `marketing`, `slogan`, `meme`, `poem`, `lyrics`, `blog` or `seo` |
| `--keywords` | Main keyword(s) for SEO articles |
| `--versions` | Number of versions, default 3 |
| `--model` | Model name, default `gpt-4o-mini` |
| `--words` | Target length in words (soft limit, given to the model) |
| `--max-tokens` | Maximum generated tokens (hard limit), default 600 |
| `--temperature` | Randomness, 0–2 |
| `--top-p` | Nucleus sampling, 0–1 |
| `--presence-penalty` | -2 to 2, encourages new topics |
| `--frequency-penalty` | -2 to 2, discourages repeated words |
| `--reasoning-effort` | `minimal`, `low`, `medium` or `high`, for reasoning models |

Generation controls are only sent when you set them. Support depends on the model: if a model rejects a setting, the program says which one.

## Model used for testing

Tested with **Gemini 3.5 Flash Lite** through Google's OpenAI-compatible endpoint, using the OpenAI Python library. The OpenAI connection itself was also tested and handled a "no credits remaining" error correctly.

## Findings from testing

- **Temperature 0.2** made the output nearly identical across versions; the default and 1.8 gave three different results, and 1.8 stayed coherent.
- **top_p 0.1** narrowed the vocabulary but did not make versions identical.
- **Frequency and presence penalties** were rejected by Gemini 3.5 Flash Lite.
- **High reasoning effort** roughly doubled the latency and made poems more culturally specific and more structured.
- **SEO articles** used the main keyword only a few times and relied on related terms.
