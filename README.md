[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://stand-with-ukraine.pp.ua)

# llm

Implements a class named [`AI::LLM`](https://github.com/crawlserv/llm/blob/main/src/AI/LLM.hpp) and a struct named [`Struct::LLMData`](https://github.com/crawlserv/llm/blob/main/src/Struct/LLMData.hpp) for using large language models (LLMs) via APIs such as OpenAI's.

Uses the `crawlservpp` namespace for compatibility with [crawlservpp](https://github.com/crawlserv/crawlservpp).

It also uses the following classes and namespaces from the main project:
* `Helper::Json` (simplified) for JSON parsing)
* `Helper::Memory` for memory management
* `Helper::Strings` for string handling (requires `boost::algorithm`)
* `Main::Exception` for exception handling
* `Wrapper::Curl` for networking (requires `libcurl`)
* `Wrapper::CurlList` for HTTP headers (requires `libcurl`)

It includes the following external libraries:
* `date.h`
* `rapidjson`

Supports multi-threading.

For the [test program](https://github.com/crawlserv/llm/blob/main/src/main.cpp), create a `config` file with the following entry:

```
key=[your API key]
model=[the selected large-language model]
prompt=[the prompt to perform]
max=[maximum number of tokens to be returned]
```

Optional entries include `org=...` and `proj=...` for specific organizations or projects.

Lines in `config` starting with `#` will be ignored.
