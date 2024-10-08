#include "AI/LLM.hpp"
#include "Main/Exception.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
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

void progress(float value) {
	static std::mutex lock;

	std::lock_guard<std::mutex> locked(lock);

	std::cout << "\r";

	if(value < 1.f - std::numeric_limits<float>::epsilon()) {
		std::cout << ' ';
	}

	if(value < 0.1f - std::numeric_limits<float>::epsilon()) {
		std::cout << ' ';
	}

	std::cout << std::fixed << std::setprecision(1) << value * 100 << "%" << std::flush;
}

int main(int /*argc*/, char* /*argv*/[]) {
	// set up instance for accessing LLM API
	const Config config("config");
	const auto organization{config.get("org")};
	const auto project{config.get("proj")};
	std::vector<std::string> httpHeaders;

	addHeaderIfNotEmpty(httpHeaders, "OpenAI-Organization: ", organization);
	addHeaderIfNotEmpty(httpHeaders, "OpenAI-Project: ", project);

	crawlservpp::AI::LLM llm("https://api.openai.com/v1/", config.get("key"), httpHeaders);

	llm.setModel(config.get("model"));
	llm.setPrompt(config.get("prompt"));
	llm.setProgressCallback(progress);

	if(!config.get("max").empty()) {
		llm.setMaxTokens(strtoul(config.get("max").c_str(), nullptr, 10));
	}

	// print available models
	std::size_t modelCounter{};

	for(const auto& model : llm.listModels()) {
		++modelCounter;

		std::cout << "[" << modelCounter << "] " << model << "\n";
	}

	// collect texts
	const std::string path{"inputs"};
	std::vector<std::string> inputs;

	for(const auto& entry : std::filesystem::directory_iterator(path)) {
		if(entry.is_regular_file() && entry.path().extension() == ".txt") {
			std::ifstream in(entry.path());

			if(!in.is_open()) {
				std::cout << "Could not read: " << entry.path() << "\n";

				continue;
			}

			std::string line;
			std::string content;

			while(std::getline(in, line)) {
				content += line;

				content.push_back('\n');
			}

			if(!content.empty()) {
				content.pop_back();
			}

			inputs.emplace_back(content);
		}
	}

	llm.addTexts(inputs);

	llm.run();

	std::cout << "\n";

	const auto results{llm.getResults()};
	std::size_t resultCounter{};

	llm.free();

	for(const auto& result : results) {
		++resultCounter;

		std::cout << "[" << resultCounter << "] " << result << "\n";
	}

	return EXIT_SUCCESS;
}
