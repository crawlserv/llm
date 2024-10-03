/*
 *
 * ---
 *
 *  Copyright (C) 2019–2024 Anselm Schmidt
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version in addition to the terms of any
 *  licences already herein identified.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * ---
 *
 * LLM.hpp
 *
 * Class for using large language models via APIs such as OpenAI's.
 *
 *  Created on:  Oct 2, 2024
 *      Author: ans
 */

#ifndef AI_LLM_HPP_
#define AI_LLM_HPP_

#include "../Helper/Json.hpp"
#include "../Helper/Memory.hpp"
#include "../Main/Exception.hpp"
#include "../Timer/Simple.hpp"
#include "../Wrapper/Curl.hpp"
#include "../Wrapper/CurlList.hpp"

#include <atomic>	// std::atomic
#include <cstddef>	// std::size_t
#include <mutex>	// std::lock_guard, std::mutex
#include <string>	// std::string
#include <thread>	// std::thread
#include <vector>	// std::vector

//DEBUG
#include <iostream>

//! Namespace for AI functionality.
namespace crawlservpp::AI {

	/*
	 * CONSTANTS
	 */

	///@name Constants
	///@{

	//! Approximate number of tokens in a text by default.
	constexpr auto defaultTokensPerCharacter{0.3f};

	//! Number of milliseconds to sleep between cycles.
	constexpr auto msSleepInBetween{1};

	///@}

	/*
	 * DECLARATION
	 */
	
	//! Class for using large language models (LLMs) via APIs such as OpenAI's.
	class LLM {
	public:
		///@name Constructor
		///@{

		LLM(const std::string& urlEndPoint, const std::string& apiKey, const std::vector<std::string>& headers);

		///@}
		///@name Settings
		///@{

		void setModel(const std::string& model);
		void setTokensPerCharacter(float ratio);
		void setPrompt(const std::string& prompt);
		void setMaxTokens(std::size_t numTokens);
		void setMaxThreads(std::size_t numThreads);

		///@}
		///@name Requests
		///@{

		[[nodiscard]] std::vector<std::string> listModels() const;
		void addInput(const std::string& input);
		void addInputs(const std::vector<std::string>& inputs);
		void run();

		///@}
		///@name Results
		///@{

		[[nodiscard]] bool isReady() const;
		[[nodiscard]] std::vector<std::string> getResults() const;

		///@}
		///@name Memory
		///@{
		
		void free();
		
		///@}

		/*
		 * CLASS FOR LLM EXCEPTIONS
		 */

		//! Class for LLM exceptions.
		/*!
		 * This exception is being thrown when
		 * - settings are changed while LLMs are running
		 *    (i.e., between @ref run() and @ref free())
		 */
		MAIN_EXCEPTION_CLASS();

	private:
		bool isRunning{false};
		float tokensPerCharacter{AI::defaultTokensPerCharacter};

		std::size_t maxTokens{};
		std::size_t maxThreads{};
		std::size_t remainingRequests{};
		std::size_t remainingTokens{};

		std::string url;
		std::string key;
		std::string currentModel;
		std::string currentPrompt;

		std::atomic<bool> isInitFinished{false};

		std::vector<std::string> headers;
		std::vector<std::string> models;
		std::vector<std::string> texts;
		std::vector<std::string> results;
		std::vector<std::thread> threads;
		std::vector<bool> isThreadsFinished;

		Timer::Simple requestTimer;	// time of last request sent
		Timer::Simple tokenTimer;	// time of last token received

		mutable std::mutex settingsLock;
		mutable std::mutex textsLock;
		mutable std::mutex requestTimerLock;
		mutable std::mutex tokenTimerLock;
		mutable std::mutex resultsLock;
		mutable std::mutex finishedLock;

		std::thread initThread;

		// private helper functions
		void notRunning(const std::string& action);
		void threadFunction(std::size_t textIndex, std::size_t threadIndex);

		// private static helper functions
		[[nodiscard]] rapidjson::Document apiRequest(
				const std::string& url,
				const std::string& key,
				const std::vector<std::string>& headers,
				const std::string& json
		);
		[[nodiscard]] static std::size_t writeCallback(
				char * content,
				std::size_t size,
				std::size_t nMemB,
				std::string * ptrBuffer
		);

	}; /* class LLM */

	/*
	 * IMPLEMENTATION
	 */

	//! Constructor.
	/*!
	 * Sets global settings for accessing the API.
	 *
	 * \param endPoint The URL of the API endpoint to be used.
	 * 
	 * \param apiKey The key to be used with every API request.
	 * 
	 * \param httpHeaders Additional HTTP headers for each request
	 *   (or an empty std::vector).
	 */
	LLM::LLM(const std::string& urlEndPoint, const std::string& apiKey, const std::vector<std::string>& httpHeaders)
	 : url(urlEndPoint), key(apiKey), headers(httpHeaders) {
		const auto models{LLM::apiRequest(this->url + "models", this->key, this->headers, {})};

		//TODO DEBUG
		std::cout << Helper::Json::stringify(models) << std::endl;
	}

	//! Sets the current large language model to use.
	/*!
	 * \param model The name of the model.
	 * 
	 * To retrieve a (possibly cached) list of models from the API,
	 *  use @ref listModels().
	 * 
	 * \throws LLM::Exception if set after calling @ref run()
	 *   and before calling @ref free().
	 */
	void LLM::setModel(const std::string& model) {
		this->notRunning("set model");

		this->currentModel = model;
	}

	//! Sets the approximate ratio of tokens per character.
	/*!
	 * \param ratio The ratio of tokens per character.
	 *   The default value is @ref defaultTokensPerCharacter.
	 * 
	 * \throws LLM::Exception if set after calling @ref run()
	 *   and before calling @ref free().
	 */
	void LLM::setTokensPerCharacter(float ratio) {
		this->tokensPerCharacter = ratio;
	}

	//! Sets the current prompt to use.
	/*!
	 * \param prompt The prompt.
	 * 
	 * The prompt will be sent together with each input.
	 *  If empty, the corresponding option will not be included
	 *  in API requests.
	 * 
	 * \throws LLM::Exception if set after calling @ref run()
	 *   and before calling @ref free().
	 */
	void LLM::setPrompt(const std::string& prompt) {
		this->notRunning("set prompt");

		this->currentPrompt = prompt;
	}

	//!

	//! Sets the maximum number of tokens to be returned.
	/*!
	 * \param numTokens The maximum number of tokens.
	 *   If zero, the corresponding option will not be included
	 *   in API requests.
	 * 
	 * \throws LLM::Exception if set after calling @ref run()
	 *   and before calling @ref free().
	 */
	void LLM::setMaxTokens(std::size_t numTokens) {
		this->notRunning("set number of tokens");

		this->maxTokens = numTokens;
	}

	//! Sets the maximum number of threads to be used.
	/*!
	 * \param numThreads The maximum number of threads used
	 *  for API requests. If zero, an approximated number
	 *  of (virtual) CPU cores will be used.
	 * 
	 * \throws LLM::Exception if set after calling @ref run()
	 *   and before calling @ref free().
	 */
	void LLM::setMaxThreads(std::size_t numThreads) {
		this->notRunning("set number of threads");

		this->maxThreads = numThreads;
	}

	void LLM::run() {
		this->isRunning = true;

		this->results = std::vector<std::string>(this->texts.size());
		this->threads = std::vector<std::thread>(this->maxThreads);
		this->isThreadsFinished = std::vector<bool>(this->maxThreads, false);

		//TODO
	}

	/*
	 * PRIVATE HELPER FUNCTIONS
	 */

	// throws an exception if the given action is requested between run() and free()
	void LLM::notRunning(const std::string& action) {
		if(this->isRunning) {
			throw LLM::Exception("Cannot " + action + " while LLMs are running");
		}
	}

	// thread function performing a request on a specific text
	void LLM::threadFunction(std::size_t textIndex, std::size_t threadIndex) {
		// copy options into text
		std::string endPoint;
		std::string apiKey;
		std::string model;
		std::string prompt;
		std::vector<std::string> httpHeaders;

		{
			std::lock_guard<std::mutex> settingsLocked(this->settingsLock);

			endPoint = this->url;
			apiKey = this->key;
			model = this->currentModel;
			prompt = this->currentPrompt;
			httpHeaders = this->headers;
		}

		// copy text into thread
		std::string text;

		{
			std::lock_guard<std::mutex> textsLocked(this->textsLock);

			text = this->texts.at(textIndex);
		}

		// construct JSON request
		std::string request;

		//TODO

		// perform request
		const auto jsonReply{apiRequest(endPoint, apiKey, httpHeaders, request)};

		// parse resulting JSON
		std::string result;

		//TODO

		// save result
		{
			std::lock_guard<std::mutex> resultsLocked(this->resultsLock);

			this->results.at(textIndex) = result;
		}

		// finish thread
		{
			std::lock_guard<std::mutex> finishedLocked(this->finishedLock);

			this->isThreadsFinished.at(threadIndex) = true;
		}
	}

	/*
	 * PRIVATE STATIC HELPER FUNCTIONS
	 */

	// sends an API request to the endpoint, blocks until the request is finished
	rapidjson::Document LLM::apiRequest(
			const std::string& url,
			const std::string& key,
			const std::vector<std::string>& headers,
			const std::string& json
	) {
		// initialize networking
		Wrapper::Curl curl;
		std::string buffer;

		curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, LLM::writeCallback);
		curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &buffer);
		curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
		curl_easy_setopt(curl.get(), CURLOPT_VERBOSE, 1L);

		// set headers
		Wrapper::CurlList headerList;

		headerList.append("Authorization: Bearer " + key);
		headerList.append(headers);

		curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headerList.get());

		// set json
		if(!json.empty()) {
			//TODO
		}

		// perform request
		const auto result{curl_easy_perform(curl.get())};

		// check for error
		if(result != CURLE_OK) {
			throw LLM::Exception(curl_easy_strerror(result));
		}

		auto jsonReply{Helper::Json::parseRapid(buffer)};

		//TODO: check for error in JSON

		return jsonReply;
	}

	// write function for libcurl
	std::size_t LLM::writeCallback(
		char * content,
		std::size_t size,
		std::size_t nMemB,
		std::string * ptrBuffer
	) {
		const auto total{size * nMemB};

		ptrBuffer->append(content, total);

		return total;
	}

} /* namespace crawlservpp::AI */

#endif /* AI_LLM_HPP_ */
