#pragma once
#include <vector>
#include <string>
#include <numeric>

namespace translation {
namespace utils{

std::vector<std::string> splitString(const std::string &str, char delim);

std::vector<std::string> getLines(const std::string &str);

std::string indentAllLines(const std::string &str);

template<class T>
T interleave(const std::vector<T> &elements,
             T interText) {

  if(elements.size() == 0){
    return T{};
  }

  if(elements.size() == 1){
    return elements[0];
  }

  auto interLam = [&](const T &lhs, const T &rhs) {
    return lhs + interText + rhs;
  };
  return std::accumulate(std::next(elements.begin()), elements.end(),
                         elements[0], interLam);
}

bool isWhitespace(unsigned char);

}
}