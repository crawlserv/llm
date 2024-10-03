# llm

Implements a class names `AI::LLM` for using large language models (LLMs) via APIs such as OpenAI's.

Uses the `crawlservpp` namespace for compatibility with [crawlservpp](https://github.com/crawlserv/crawlservpp crawlservpp).

It also uses the following classes and namespaces from the main project:
* `Helper::DateTime` for time management
* `Helper::Json` (simplified) for JSON parsing)
* `Helper::Memory` for memory management
* `Helper::Strings` for string handling (requires `boost::algorithm`)
* `Main::Exception` for exception handling
* `Timer::Simple` for simple timing.
* `Wrapper::Curl` for networking (requires `libcurl`)
* `Wrapper::CurlList` for HTTP headers (requires `libcurl`)

It includes the following external libraries:
* `date.h`
* `rapidjson`

Supports multi-threading.

For the test program, create a `config` file with the following entry:

`key=[your API key]`

Optional entries include `org` and `proj` for specific organizations or projects.
