#include "AI/LLM.hpp"
#include "Main/Exception.hpp"

#include <cstddef>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

class Config {
	using Entries = std::unordered_map<std::string, std::string>;
public:
	Config(const std::string& fileName) {
		std::ifstream in(fileName.c_str());
		std::string line;

		if(!in.is_open()) {
			throw crawlservpp::Main::Exception("Could not open '" + fileName + "'");
		}

		while(std::getline(in, line)) {
			if(line.empty() || line[0] == '#') {
				continue;
			}

			const auto separator{line.find('=')};

			if(separator == std::string::npos) {
				this->entries.emplace(line, std::string{});

				continue;
			}

			this->entries.emplace(line.substr(0, separator), line.substr(separator + 1));
		}

		in.close();
	}	

	[[nodiscard]] std::string get(const std::string& key) const {
		const auto it{this->entries.find(key)};

		if(it == this->entries.cend()) {
			return {};
		}

		return it->second;
	}

private:
	Entries entries;
};

void addHeaderIfNotEmpty(std::vector<std::string>& to, const std::string& name, const std::string& value) {
	if(value.empty()) {
		return;
	}

	to.emplace_back(name + ": " + value);
}

int main(int /*argc*/, char* /*argv*/[]) {
	const Config config("config");
	const auto organization{config.get("org")};
	const auto project{config.get("proj")};
	std::vector<std::string> httpHeaders;

	addHeaderIfNotEmpty(httpHeaders, "OpenAI-Organization: ", organization);
	addHeaderIfNotEmpty(httpHeaders, "OpenAI-Project: ", project);

	crawlservpp::AI::LLM llm("https://api.openai.com/v1/", config.get("key"), httpHeaders);

	return EXIT_SUCCESS;
}
