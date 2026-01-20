//
// Created by chris on 1/20/26.
//

#ifndef ITERATEDFUNCTIONS_UICALLBACK_HPP
#define ITERATEDFUNCTIONS_UICALLBACK_HPP

#include <functional>
#include <variant>
#include <string>

#include "Common.hpp"

namespace ifs
{

/**
 * @brief Callback for continuous (float) UI parameters
 */
struct ContinuousCallback
{
	std::function<void(float)> setter;
	std::function<float()> getter;
	float min;
	float max;
	bool logarithmic = false;  // Optional: use logarithmic scale in UI
};

/**
 * @brief Callback for discrete (int) UI parameters
 */
struct DiscreteCallback
{
	std::function<void(int)> setter;
	std::function<int()> getter;
	int min;
	int max;
};

/**
 * @brief Callback for toggle (bool) UI parameters
 */
struct ToggleCallback
{
	std::function<void(bool)> setter;
	std::function<bool()> getter;
};

enum class CallbackType
{
	Continuous,
	Discrete,
	Toggle
};

/**
 * @brief Generic UI callback that can hold continuous, discrete, or toggle callbacks
 *
 * Frontends and backends return vectors of these to expose UI controls generically.
 * The UI rendering code interprets the callback type and renders appropriate ImGui widgets.
 */
struct UICallback
{
	std::string field_name;
	std::variant<ContinuousCallback, DiscreteCallback, ToggleCallback> callback;

	UICallback(std::string name, ContinuousCallback cb)
		: field_name(std::move(name)), callback(std::move(cb)) {}

	UICallback(std::string name, DiscreteCallback cb)
		: field_name(std::move(name)), callback(std::move(cb)) {}

	UICallback(std::string name, ToggleCallback cb)
		: field_name(std::move(name)), callback(std::move(cb)) {}

	[[nodiscard]] CallbackType get_callback_type() const
	{
		return std::visit(
			overloaded{
				[](const ContinuousCallback&) { return CallbackType::Continuous; },
				[](const DiscreteCallback&) { return CallbackType::Discrete; },
				[](const ToggleCallback&) { return CallbackType::Toggle; },
			},
			callback);
	}

	// Getters for specific callback types
	[[nodiscard]] const ContinuousCallback* as_continuous() const {
		return std::get_if<ContinuousCallback>(&callback);
	}

	[[nodiscard]] const DiscreteCallback* as_discrete() const {
		return std::get_if<DiscreteCallback>(&callback);
	}

	[[nodiscard]] const ToggleCallback* as_toggle() const {
		return std::get_if<ToggleCallback>(&callback);
	}
};

} // namespace ifs

#endif // ITERATEDFUNCTIONS_UICALLBACK_HPP
