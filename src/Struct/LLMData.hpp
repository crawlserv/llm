/*
 *
 * ---
 *
 *  Copyright (C) 2024 Anselm Schmidt (ans[ät]ohai.su)
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
 * LLMData.hpp
 *
 * Data for a thread processing a text via a LLM.
 *
 *  Created on: Oct 7, 2024
 *      Author: ans
 */

#ifndef STRUCT_LLMDATA_HPP_
#define STRUCT_LLMDATA_HPP_

#include <cstddef>	// std::size_t
#include <string>	// std::string
#include <vector>	// std::vector

namespace crawlservpp::Struct {

	//! A structure containing data for a thread processing a text via a LLM.
	/*!
	 * Contains the API end point, key, model, prompt, maximum number of tokens,
	 *  http headers, and the text to be processed.
	 */
	struct LLMData {
		//@name Properties
		///@{

		//! The maximum number of tokens to be returned.
		/*!
		 * A value of zero means that no token limit is applied.
		 */
		std::size_t maxTokens{};

		//! URL to the endpoint of the LLM API.
		std::string endPoint;

		//! The API key to be used.
		std::string apiKey;

		//! The LLM to be used.
		std::string model;

		//! The prompt for using the LLM.
		std::string prompt;

		//! The text to be processed by the current thread.
		std::string text;

		//! The HTTP headers to be sent on API requests.
		std::vector<std::string> httpHeaders;

		///@}
		//@name Constructor
		///@{

		//! Constructor setting endpoint and API key
		/*!
		 * \param url URL of the API endpoint.
		 *
		 * \param key API key to be used.
		 */
		LLMData(const std::string& url, const std::string& key) : endPoint(url), apiKey(key) {}

		///@}
	};

} /* namespace crawlservpp::Struct */

#endif /* STRUCT_LLMDATA_HPP_ */