/*
 *
 * ---
 *
 *  Copyright (C) 2024 Anselm Schmidt
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
#include "../Struct/LLMData.hpp"
#include "../Wrapper/Curl.hpp"
#include "../Wrapper/CurlList.hpp"

#include <algorithm>	// std::transform
#include <atomic>		// std::atomic
#include <chrono>		// std::chrono::*
#include <cmath>		// std::lround
#include <cstddef>		// std::size_t
#include <exception>	// std::exception
#include <functional>	// std::bind, std::function
#include <limits>		// std::numeric_limits
#include <mutex>		// std::lock_guard, std::mutex
#include <string>		// std::string
#include <thread>		// std::this_thread::sleep_for, std::thread
#include <vector>		// std::vector

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
	constexpr auto msSleepInBetween{10};

	//! Number of milliseconds to sleep when the token or request per minute limit is reached.
	constexpr auto msSleepOnLimit{100};

	///@}

	/*
	 * DECLARATION
	 */
	
	//! Class for using large language models (LLMs) via APIs such as OpenAI's.
	class LLM {
		using ProgressCallback = std::function<void(float)>;
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
		void setProgressCallback(const ProgressCallback& callback);

		///@}
		///@name Requests
		///@{

		[[nodiscard]] std::vector<std::string> listModels() const;
		void addText(const std::string& text);
		void addTexts(const std::vector<std::string>& texts);
		void run();

		///@}
		///@name Results
		///@{

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
		 * - the list of LLMs cannot be retrieved
		 * - no model was selected
		 * - selected model is an empty string or not available
		 * - the API returns an error
		 * - the response from the API cannot be parsed
		 */
		MAIN_EXCEPTION_CLASS();

	private:
		bool isRequestLimitReset{false};
		bool isTokenLimitReset{false};

		float tokensPerCharacter{AI::defaultTokensPerCharacter};

		std::size_t maxTokens{};
		std::size_t maxThreads{};
		std::size_t requestLimit{std::numeric_limits<std::size_t>::max()};
		std::size_t tokenLimit{std::numeric_limits<std::size_t>::max()};
		std::size_t requestsRemaining{std::numeric_limits<std::size_t>::max()};
		std::size_t tokensRemaining{std::numeric_limits<std::size_t>::max()};
		std::size_t requestsMade{};
		std::size_t tokensSent{};

		std::string url;
		std::string key;
		std::string currentModel;
		std::string currentPrompt;

		std::atomic<std::size_t> textsDone{};

		std::vector<std::string> headers;
		std::vector<std::string> models;
		std::vector<std::string> inputs;
		std::vector<std::string> results;
		std::vector<std::thread> threads;
		std::vector<bool> isThreadsUsed;
		std::vector<bool> isThreadsFinished;

		std::chrono::time_point<std::chrono::steady_clock> requestLimitTimeout;
		std::chrono::time_point<std::chrono::steady_clock> tokenLimitTimeout;

		mutable std::mutex settingsLock;
		mutable std::mutex limitsLock;
		mutable std::mutex resultsLock;
		mutable std::mutex finishedLock;

		std::thread initThread;

		ProgressCallback progressCallback;

		// private helper functions
		void modelsReceived(const rapidjson::Document& json);
		void loop();
		void finish();
		void assignText(std::size_t textIndex);
		void assignThread(std::size_t textIndex, std::size_t threadIndex);
		void calculateMaxThreads();
		void threadFunction(std::size_t textIndex, std::size_t threadIndex);
		[[nodiscard]] Struct::LLMData copySettingsToThread() const;
		void checkLimits(std::size_t textLength);
		[[nodiscard]] rapidjson::Document apiRequest(
				const std::string& command,
				const Struct::LLMData& data
		);
		void limitsReceived(
			std::size_t limitRequests,
			std::size_t limitTokens,
			std::size_t remainingRequests,
			std::size_t remainingTokens,
			std::size_t resetRequests,
			std::size_t resetTokens
		);
		void textDone();

		// private static helper functions
		static void jsonHasMember(
				const rapidjson::Document& jsonObject,
				const std::string& jsonString,
				const std::string& name,
				const std::string& action
		);
		[[nodiscard]] static std::string jsonRequest(const Struct::LLMData& src);
		static void parseHeader(
				const std::string& header,
				const std::string& name,
				std::size_t& to,
				bool& isFoundTo,
				bool isParseTime
		);
		[[nodiscard]] static std::size_t parseTimeMs(const std::string& src);
		[[nodiscard]] static std::size_t headerCallback(
				char * content,
				std::size_t size,
				std::size_t nMemB,
				std::vector<std::string> * ptrHeaders
		);
		[[nodiscard]] static std::size_t writeCallback(
				char * content,
				std::size_t size,
				std::size_t nMemB,
				std::string * ptrBuffer
		);
		[[nodiscard]] static std::size_t toUL(const std::string& src); 

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
	 * 
	 * \throws LLM::Exception if list of models could not be received from the API.
	 */
	LLM::LLM(const std::string& urlEndPoint, const std::string& apiKey, const std::vector<std::string>& httpHeaders)
	 : url(urlEndPoint), key(apiKey), headers(httpHeaders) {
	 	// get models from the API
	 	const Struct::LLMData data(this->url, this->key);

		this->modelsReceived(this->apiRequest("models", data));
	}

	//! Sets the current large language model to use.
	/*!
	 * \param model The name of the model.
	 * 
	 * To retrieve a (possibly cached) list of models from the API,
	 *  use @ref listModels().
	 * 
	 * \throws LLM::Exception if given model is empty or not available.
	 */
	void LLM::setModel(const std::string& model) {
		if(model.empty()) {
			throw LLM::Exception("No model selected");
		}

		for(const auto& availableModel : this->models) {
			if(availableModel == model) {
				this->currentModel = model;

				return;
			}
		}

		throw LLM::Exception("Model \"" + model + "\" is not available");
	}

	//! Sets the approximate ratio of tokens per character.
	/*!
	 * \param ratio The ratio of tokens per character.
	 *   The default value is @ref defaultTokensPerCharacter.
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
	 */
	void LLM::setPrompt(const std::string& prompt) {
		this->currentPrompt = prompt;
	}

	//!

	//! Sets the maximum number of tokens to be returned.
	/*!
	 * \param numTokens The maximum number of tokens.
	 *   If zero, the corresponding option will not be included
	 *   in API requests.
	 */
	void LLM::setMaxTokens(std::size_t numTokens) {
		this->maxTokens = numTokens;
	}

	//! Sets the maximum number of threads to be used.
	/*!
	 * \param numThreads The maximum number of threads used
	 *  for API requests. If zero, an approximated number
	 *  of (virtual) CPU cores will be used.
	 */
	void LLM::setMaxThreads(std::size_t numThreads) {
		this->maxThreads = numThreads;
	}

	//! Sets the callback for progress notifications.
	/*!
	 * The given callback will be called for each processed text.
	 * 
	 * \param callback `std::function` binding to be called on progress.
	 *   The function itself should be of the format `void x(float)`. 
	 * 
	 * \note Make sure to write your callback in a thread-safe manner,
	 *  as multiple callbacks might be called at the same time.
	 */
	void LLM::setProgressCallback(const ProgressCallback& callback) {
		this->progressCallback = callback;
	}

	//! Adds a text for processing.
	/*!
	 * \param text The text to be added to the inputs.
	 */
	void LLM::addText(const std::string& text) {
		this->inputs.emplace_back(text);
	}

	//! Adds texts for processing.
	/*!
	 * \param texts A vector of texts to be added to the inputs.
	 */
	void LLM::addTexts(const std::vector<std::string>& texts) {
		this->inputs.reserve(this->inputs.size() + texts.size());

		this->inputs.insert(this->inputs.end(), texts.cbegin(), texts.cend());
	}

	//! Lists all available models retrieved from the API.
	/*!
	 * \returns A vector containing the ids of all available models.
	 */
	std::vector<std::string> LLM::listModels() const {
		return this->models;
	}

	//! Runs the prompt on all texts using the API and multi-threading.
	/*!
	 * \throws LLM::Exception if no model has been selected.
	 */
	void LLM::run() {
		if(this->currentModel.empty()) {
			throw LLM::Exception("No model has been selected");
		}

		this->calculateMaxThreads();

		this->results = std::vector<std::string>(this->inputs.size());
		this->threads = std::vector<std::thread>(this->maxThreads);
		this->isThreadsFinished = std::vector<bool>(this->maxThreads, false);
		this->isThreadsUsed = std::vector<bool>(this->maxThreads, false);

		this->loop();
		this->finish();
	}

	//! Gets the results.
	/*!
	 * \returns A vector containing all the results,
	 *   in the order of the added texts.
	 */
	std::vector<std::string> LLM::getResults() const {
		return this->results;
	}

	//! Frees memory.
	/*!
	 * Discards all memory allocated in @ref run().
	 */
	void LLM::free() {
		Helper::Memory::free(this->results);
		Helper::Memory::free(this->threads);
		Helper::Memory::free(this->isThreadsFinished);
		Helper::Memory::free(this->isThreadsUsed);
	}

	/*
	 * PRIVATE HELPER FUNCTIONS
	 */

	// parse JSON response containing available LLMs
	void LLM::modelsReceived(const rapidjson::Document& json) {
		std::string jsonString;

		// parse JSON response
		try {
			jsonString = Helper::Json::stringify(json);
		}
		catch(const Helper::Json::Exception& e) {
			throw LLM::Exception(
					"Cannot retrieve models from the API: "
					+ std::string{e.what()}	
			);
		}

		if(!json.IsObject()) {
			throw LLM::Exception("Cannot retrieve models from the API: JSON response is not an object – " + jsonString);
		}

		LLM::jsonHasMember(json, jsonString, "object", "retrieve models from the API");
		LLM::jsonHasMember(json, jsonString, "data", "retrieve models from the API");

		if(!json["data"].IsArray()) {
			throw LLM::Exception("Cannot retrieve models from the API: \"data\" is not an array – " + jsonString);
		}

		const auto& dataArray{json["data"]};

		this->models.reserve(dataArray.Size());

		for(const auto& model : dataArray.GetArray()) {
			if(!model.IsObject()) {
				throw LLM::Exception(
						"Cannot retrieve models from the API: An entry in \"data\" is not an object – "
						+ jsonString
				);
			}

			if(!model.HasMember("id")) {
				throw LLM::Exception(
						"Cannot retrieve models from the API: An entry in \"data\" has no \"id\" – "
						+ jsonString
				);
			}

			if(!model["id"].IsString()) {
				throw LLM::Exception(
						"Cannot retrieve models from the API: An entry's \"id\" in \"data\" is not string – "
						+ jsonString
				);
			}

			this->models.push_back(model["id"].GetString());
		}
	}

	// main loop assigning texts to threads
	void LLM::loop() {
		const auto numTexts{this->inputs.size()};

		for(std::size_t textIndex{}; textIndex < numTexts; ++textIndex) {
			this->assignText(textIndex);

			std::this_thread::sleep_for(std::chrono::milliseconds(msSleepInBetween));
		}
	}

	// waiting for threads to finish
	void LLM::finish() {
		for(std::size_t threadIndex{}; threadIndex < this->threads.size(); ++threadIndex) {
			if(this->isThreadsUsed.at(threadIndex)) {
				this->threads.at(threadIndex).join();
				this->isThreadsUsed.at(threadIndex) = false;
			}
		}
	}

	// assign one particular text to a thread
	void LLM::assignText(std::size_t textIndex) {
		const auto numThreads{this->threads.size()};
		bool isNotAssigned{true};

		do {
			for(std::size_t threadIndex{}; threadIndex < numThreads; ++threadIndex) {
				if(!(this->isThreadsUsed.at(threadIndex))) {
					this->isThreadsUsed.at(threadIndex) = true;

					this->assignThread(textIndex, threadIndex);

					isNotAssigned = false;

					break;
				}
				
				{
					std::lock_guard<std::mutex> finishedLocked(this->finishedLock);

					isNotAssigned = !(this->isThreadsFinished.at(threadIndex));
				}

				if(isNotAssigned) {
					continue; // thread is still busy
				}

				this->threads.at(threadIndex).join();

				this->assignThread(textIndex, threadIndex);

				break;
			}
		} while(isNotAssigned);
	}

	void LLM::assignThread(std::size_t textIndex, std::size_t threadIndex) {
		this->threads.at(threadIndex) = std::thread(&LLM::threadFunction, this, textIndex, threadIndex);
	}

	// calculates the number of threads to be used
	void LLM::calculateMaxThreads() {
		if(this->maxThreads == 0) {
			this->maxThreads = std::thread::hardware_concurrency();
		}
		
		if(this->maxThreads == 0) {
			this->maxThreads = 1;
		}
	}

	// thread function performing a request on a specific text
	void LLM::threadFunction(std::size_t textIndex, std::size_t threadIndex) {
		// copy data into text
		auto data{this->copySettingsToThread()};

		data.text = this->inputs.at(textIndex);

		// check limits
		this->checkLimits(data.text.size());

		// perform API request
		const auto jsonReply{this->apiRequest("chat/completions", data)};

		// check resulting JSON
		if(!jsonReply.HasMember("choices") || !jsonReply["choices"].IsArray() || jsonReply["choices"].Empty()) {
			throw LLM::Exception(
					"Could not parse result: no \"choices\" given – "
					+ Helper::Json::stringify(jsonReply)
			);
		}

		if(!jsonReply["choices"][0].HasMember("message") || !jsonReply["choices"][0]["message"].IsObject()) {
			throw LLM::Exception(
					"Could not parse result: first choice has no or invalid \"message\" – "
					+ Helper::Json::stringify(jsonReply)
			);
		}

		if(
				!jsonReply["choices"][0]["message"].HasMember("content")
				|| !jsonReply["choices"][0]["message"]["content"].IsString()
		) {
			throw LLM::Exception(
					"Could not parse result: first choice has no or invalid \"message\".\"content\" – "
					+ Helper::Json::stringify(jsonReply)
			);
		}

		// save result
		const auto result{jsonReply["choices"][0]["message"]["content"].GetString()};

		{
			std::lock_guard<std::mutex> resultsLocked(this->resultsLock);

			this->results.at(textIndex) = result;
		}

		// finish thread
		{
			std::lock_guard<std::mutex> finishedLocked(this->finishedLock);

			this->isThreadsFinished.at(threadIndex) = true;
		}

		// notify about progress (one more text done)
		this->textDone();
	}

	// checks the current limits and sleeps until they are reset
	void LLM::checkLimits(std::size_t textLength) {
		const auto tokens{textLength * this->tokensPerCharacter};

		// waiting for limits to reset
		while(true) {
			{
				std::lock_guard<std::mutex> limitsLocked(this->limitsLock);

				// reset limits if out of date
				if(
						!(this->isRequestLimitReset)	
						&& this->requestLimitTimeout > std::chrono::time_point<std::chrono::steady_clock>{}
						&& this->requestLimitTimeout < std::chrono::steady_clock::now()
				) {
					this->isRequestLimitReset = true;
					this->requestsRemaining = this->requestLimit;
				}

				if(
						!(this->isTokenLimitReset)	
						&& this->tokenLimitTimeout > std::chrono::time_point<std::chrono::steady_clock>{}
						&& this->tokenLimitTimeout < std::chrono::steady_clock::now()
				) {
					this->isTokenLimitReset = true;
					this->tokensRemaining = this->tokenLimit;
				}

				// check limits
				if(this->requestsRemaining > this->requestsMade && this->tokensRemaining > tokens + this->tokensSent) {
					--(this->requestsRemaining);

					this->tokensRemaining -= tokens;

					break;
				}
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(msSleepOnLimit));
		}
	}

	// re-sets request and token limit from server
	void LLM::limitsReceived(
			std::size_t limitRequests,
			std::size_t limitTokens,
			std::size_t remainingRequests,
			std::size_t remainingTokens,
			std::size_t resetRequests,
			std::size_t resetTokens
	) {
		std::lock_guard<std::mutex> limitsLocked(this->limitsLock);

		this->requestLimit = limitRequests;
		this->tokenLimit = limitTokens;
		this->requestsRemaining = remainingRequests;
		this->tokensRemaining = remainingTokens;
		this->requestsMade = 0;
		this->tokensSent = 0;
		this->requestLimitTimeout = std::chrono::steady_clock::now() + std::chrono::milliseconds(resetRequests);
		this->tokenLimitTimeout = std::chrono::steady_clock::now() + std::chrono::milliseconds(resetTokens);
		this->isRequestLimitReset = false;
		this->isTokenLimitReset = false;
	}

	// notifies on progress
	void LLM::textDone() {
		++(this->textsDone);

		if(this->progressCallback) {
			this->progressCallback(static_cast<float>(this->textsDone) / this->inputs.size());
		}		
	}

	// copies the settings into the thread
	Struct::LLMData LLM::copySettingsToThread() const {
		std::lock_guard<std::mutex> settingsLocked(this->settingsLock);
		Struct::LLMData data(this->url, this->key);

		data.maxTokens = this->maxTokens;
		data.model = this->currentModel;
		data.prompt = this->currentPrompt;
		data.httpHeaders = this->headers;

		return data;
	}

	/*
	 * PRIVATE STATIC HELPER FUNCTIONS
	 */

	// checks whether a JSON object has a specific member
	void LLM::jsonHasMember(
			const rapidjson::Document& jsonObject,
			const std::string& jsonString,
			const std::string& name,
			const std::string& action
	) {
		if(!jsonObject.HasMember(name)) {
			throw LLM::Exception("Cannot " + action + ": JSON response has no member \"" + name + "\" – " + jsonString);
		}
	}

	// sends an API request to the endpoint, blocks until the request is finished
	rapidjson::Document LLM::apiRequest(
			const std::string& command,
			const Struct::LLMData& data
	) {
		// initialize networking
		Wrapper::Curl curl;
		const auto url{data.endPoint + command};
		std::string buffer;
		std::vector<std::string> headers;

		curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, LLM::writeCallback);
		curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, LLM::headerCallback);
		curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &buffer);
		curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &headers);
		curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
		//curl_easy_setopt(curl.get(), CURLOPT_VERBOSE, 1L);

		// set JSON request
		const auto json{LLM::jsonRequest(data)};

		if(!json.empty()) {
			curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, json.c_str());
		}

		// set headers
		Wrapper::CurlList headerList;

		headerList.append("Authorization: Bearer " + data.apiKey);

		if(!json.empty()) {
			headerList.append("Content-Type: application/json");
			headerList.append(data.httpHeaders);
		}

		curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headerList.get());

		// perform request
		const auto result{curl_easy_perform(curl.get())};

		// check for error
		if(result != CURLE_OK) {
			throw LLM::Exception(curl_easy_strerror(result));
		}

		auto jsonReply{Helper::Json::parseRapid(buffer)};

		if(jsonReply.HasMember("error")) {
			if(
					jsonReply["error"].IsObject()
					&& jsonReply["error"].HasMember("message")
					&& jsonReply["error"]["message"].IsString()
			) {
				if(jsonReply["error"].HasMember("type") && jsonReply["error"]["type"].IsString()) {
					throw LLM::Exception(
							std::string{"["}
							+ jsonReply["error"]["type"].GetString()
							+ "] "
							+ jsonReply["error"]["message"].GetString()
					);
				}

				throw LLM::Exception(jsonReply["error"]["message"].GetString());
			}

			throw("API used for large-language models returned an unknown error");
		}

		// process headers
		const std::string headerLimitRequests{"x-ratelimit-limit-requests: "};
		const std::string headerLimitTokens{"x-ratelimit-limit-tokens: "};
		const std::string headerRemainingRequests{"x-ratelimit-remaining-requests: "};
		const std::string headerRemainingTokens{"x-ratelimit-remaining-tokens: "};
		const std::string headerResetRequests{"x-ratelimit-reset-requests: "};
		const std::string headerResetTokens{"x-ratelimit-reset-tokens: "};

		std::size_t limitRequests{};
		std::size_t limitTokens{};
		std::size_t remainingRequests{};
		std::size_t remainingTokens{};
		std::size_t resetRequests{};
		std::size_t resetTokens{};

		bool isLimitRequests{false};
		bool isLimitTokens{false};
		bool isRemainingRequests{false};
		bool isRemainingTokens{false};
		bool isResetRequests{false};
		bool isResetTokens{false};

		for(const auto& header : headers) {
			LLM::parseHeader(header, headerLimitRequests, limitRequests, isLimitRequests, false);
			LLM::parseHeader(header, headerLimitTokens, limitTokens, isLimitTokens, false);
			LLM::parseHeader(header, headerRemainingRequests, remainingRequests, isRemainingRequests, false);
			LLM::parseHeader(header, headerRemainingTokens, remainingTokens, isRemainingTokens, false);
			LLM::parseHeader(header, headerResetRequests, resetRequests, isResetRequests, true);
			LLM::parseHeader(header, headerResetTokens, resetTokens, isResetTokens, true);
		}

		// update limits
		if(
				isLimitRequests
				&& isLimitTokens
				&& isRemainingRequests
				&& isRemainingTokens
				&& isResetRequests
				&& isResetTokens
		) {
			this->limitsReceived(
					limitRequests,
					limitTokens,
					remainingRequests,
					remainingTokens,
					resetRequests,
					resetTokens
			);
		}

		return jsonReply;
	}

	// constructs a JSON request for the API endpoint
	std::string LLM::jsonRequest(const Struct::LLMData& data) {
		if(data.model.empty() || data.text.empty()) {
			return {};
		}

		auto jsonObject{Helper::Json::initObject()};

		Helper::Json::addKeyValuePair(jsonObject, "model", data.model);

		if(!data.prompt.empty()) {
			const std::vector<std::pair<std::string, std::string>> promptPairs{
				{"role", "system"},
				{"content", data.prompt}
			};

			Helper::Json::addKeyValuePairs(
					jsonObject,
					"messages",
					promptPairs
			);
		}

		const std::vector<std::pair<std::string, std::string>> pairs{
			{"role", "user"},
			{"content", data.text}
		};

		Helper::Json::addKeyValuePairs(
				jsonObject,
				"messages",
				pairs
		);

		if(data.maxTokens > 0) {
			Helper::Json::addKeyValuePair(jsonObject, "max_completion_tokens", data.maxTokens);
		}

		return Helper::Json::stringify(jsonObject);
	}

	// parses a HTTP header
	void LLM::parseHeader(
			const std::string& header,
			const std::string& name,
			std::size_t& to,
			bool& isFoundTo,
			bool isParseTime
	) {
		if(header.substr(0, name.length()) != name) {
			return;
		}

		const auto& value{header.substr(name.length())};

		if(value.empty()) {
			return;
		}

		if(isParseTime) {
			to = LLM::parseTimeMs(value);
		}
		else {
			to = LLM::toUL(value);
		}

		isFoundTo = true;
	}

	// parse a time string from an API header into milliseconds
	std::size_t LLM::parseTimeMs(const std::string& src) {
	    const auto msPos{src.find("ms")};

	    if(msPos != std::string::npos) {
	    	return LLM::toUL(src.substr(0, msPos));
	    }

	    constexpr auto msPerDay{86400000};
		constexpr auto msPerHour{3600000};
		constexpr auto msPerMinute{60000};
		constexpr auto msPerSecond{1000};

	    const auto dPos{src.find('d')};
		const auto hPos{src.find('h')};
	    const auto mPos{src.find('m')};
	    const auto sPos{src.find('s')};

	    std::size_t days{};
	    std::size_t hours{};
	    std::size_t minutes{};
	    std::size_t seconds{};
	    std::size_t milliseconds{};
	    std::size_t pos{};
	    
	    if(dPos != std::string::npos) {
	    	days = LLM::toUL(src.substr(0, dPos));
	    	pos = dPos + 1;
	    }

	    if(hPos != std::string::npos) {
	    	hours = LLM::toUL(src.substr(pos, hPos));
	    	pos = hPos + 1;
	    }

	    if(mPos != std::string::npos) {
	    	minutes = LLM::toUL(src.substr(pos, mPos));
	    	pos = mPos + 1;
	    }

	    if(sPos != std::string::npos) {
	    	const auto dotPos{src.find('.', pos)};

	    	if(dotPos == std::string::npos) {
	    		seconds = LLM::toUL(src.substr(pos, sPos));
	    	}
	    	else {
	    		seconds = LLM::toUL(src.substr(pos, dotPos - pos));
	    		milliseconds = LLM::toUL(src.substr(dotPos + 1, sPos - dotPos - 1));

	    		if(milliseconds < 100) {
	    			milliseconds *= 10;
	    		}
	    	}

	    	pos = sPos + 1;
	    }

	    return days * msPerDay + hours * msPerHour + minutes * msPerMinute + seconds * msPerSecond + milliseconds;
	}

	// callback function for libcurl used for headers
	std::size_t LLM::headerCallback(
		char * content,
		std::size_t size,
		std::size_t nMemB,
		std::vector<std::string> * ptrHeaders
	) {
		if(content == nullptr || ptrHeaders == nullptr) {
			return {};
		}

		const auto total{size * nMemB};

		ptrHeaders->emplace_back(content, total);

		auto& header{ptrHeaders->back()};

		if(header.size() > 1) {
			// remove final CRLF
			header.pop_back();
			header.pop_back();
		}

		std::transform(header.begin(), header.end(), header.begin(), ::tolower);

		return total;
	}

	// callback function for libcurl used for content
	std::size_t LLM::writeCallback(
		char * content,
		std::size_t size,
		std::size_t nMemB,
		std::string * ptrBuffer
	) {
		if(content == nullptr || ptrBuffer == nullptr) {
			return {};
		}

		const auto total{size * nMemB};

		ptrBuffer->append(content, total);

		return total;
	}

	// safe conversion from string to unsigned long/std::size_t
	std::size_t LLM::toUL(const std::string& src) {
#ifdef NDEBUG
		try {
			return std::stoul(src);
		}
		catch(const std::exception& e) {
			return {};
		}
#else
		return std::stoul(src);
#endif		
	}

} /* namespace crawlservpp::AI */

#endif /* AI_LLM_HPP_ */
