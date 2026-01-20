//
// Created by chris on 1/6/26.
//

#ifndef ITERATEDFUNCTIONS_COMMON_HPP
#define ITERATEDFUNCTIONS_COMMON_HPP
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <string_view>
#include <utility>

template<class... Fs>
struct overloaded : Fs...
{
	using Fs::operator()...;
};

// Deduction guide for C++17
template<class... Fs>
overloaded(Fs...) -> overloaded<Fs...>;

#endif // ITERATEDFUNCTIONS_COMMON_HPP
